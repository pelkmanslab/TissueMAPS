# TmLibrary - TissueMAPS library for distibuted image analysis routines.
# Copyright (C) 2016  Markus D. Herrmann, University of Zurich and Robin Hafen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import os
import shutil
import random
import logging
import inspect
from copy import copy
import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.pool
import sqlalchemy.exc
from sqlalchemy.engine.url import make_url
from sqlalchemy_utils.functions import quote
from sqlalchemy.event import listens_for
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import NamedTupleCursor
from cached_property import cached_property

from tmlib.models.base import MainModel, ExperimentModel, FileSystemModel
from tmlib.models.dialect import *
from tmlib import cfg

logger = logging.getLogger(__name__)

#: Dict[str, sqlalchemy.engine.base.Engine]: mapping of chached database
#: engine objects for reuse within the current Python process hashable by URL
DATABASE_ENGINES = {}

#: int: number of pooled database connections
POOL_SIZE = 5

_SCHEMA_NAME_FORMAT_STRING = 'experiment_{experiment_id}'


def set_pool_size(n):
    '''Sets the pool size for database connections of the current Python
    process.
    '''
    logger.debug('set size of database pool to %d', n)
    global POOL_SIZE
    POOL_SIZE = n


def create_db_engine(db_uri, cache=True):
    '''Creates a database engine with a given pool size.

    Parameters
    ----------
    db_uri: str
        database uri
    cache: bool, optional
        whether engine should be cached for reuse (default: ``True``)

    Returns
    -------
    sqlalchemy.engine.base.Engine
        created database engine

    '''
    if db_uri not in DATABASE_ENGINES:
        logger.debug(
            'create database engine for process %d with pool size %d',
            os.getpid(), POOL_SIZE
        )
        if POOL_SIZE > 1:
            overflow_size = POOL_SIZE * 2
        elif POOL_SIZE == 1:
            # For parallel processes running on the cluster, we want as few
            # database connections as possible. In principle one connection
            # should be enough.
            # However, we may want to have a "Session" and a "Connection"
            # each having a connection open, simulatenously. Therefore, we
            # allow an overflow of one additional connection.
            overflow_size = 1
        else:
            raise ValueError('Pool size must be a positive integer.')
        engine = sqlalchemy.create_engine(
            db_uri, poolclass=sqlalchemy.pool.QueuePool,
            pool_size=POOL_SIZE, max_overflow=overflow_size,
        )
        if cache:
            logger.debug('cache database engine for reuse')
            DATABASE_ENGINES[db_uri] = engine
    else:
        logger.debug('reuse cached database engine for process %d', os.getpid())
        engine = DATABASE_ENGINES[db_uri]
    return engine


def _assert_db_exists(engine):
    db_url = make_url(engine.url)
    db_name = db_url.database
    try:
        logger.debug('try to connect to database "%s": %s', db_name, db_url)
        connection = engine.connect()
        connection.close()
    except sqlalchemy.exc.OperationalError as err:
        db_url = make_url(engine.url)
        db_name = db_url.database
        logger.error('could not connect to database "%s": %s', db_name, str(err))
        raise ValueError('Cannot connect to database "%s".' % db_name)


def _set_search_path(connection, schema_name):
    if schema_name is not None:
        logger.debug('set search path to schema "%s"', schema_name)
        cursor = connection.connection.cursor()
        cursor.execute('''
            SET search_path TO 'public', %(schema)s;
        ''', {
            'schema': schema_name
        })
        cursor.close()


def _customize_distributed_tables(connection, schema_name):
    cursor = connection.connection.cursor()
    tables_to_distribute_by_range = {
        'mapobjects', 'mapobject_segmentations',
        'feature_values', 'label_values'
    }
    # Change distribution method from "hash" to "range". This is important to
    # be able to target individual shards from compute nodes for parallel
    # bulk ingestion of data.
    # NOTE: database user must be given permissions to update these tables!
    for table_name in tables_to_distribute_by_range:
        cursor.execute('''
            UPDATE pg_dist_partition SET partmethod = 'r'
            WHERE logicalrelid = %(table)s::regclass
        ''', {
            'table': '{schema}.{table}'.format(
                schema=schema_name, table=table_name
            )
        })
        cursor.execute('''
            SELECT shardid FROM pg_dist_shard
            WHERE logicalrelid = %(table)s::regclass
            ORDER BY shardid;
        ''', {
            'table': '{schema}.{table}'.format(
                schema=schema_name, table=table_name
            )
        })
        shards = cursor.fetchall()
        n = len(shards)
        # NOTE: distribution column of "range" partitioned tables must have type
        # BigInteger
        total_max_value = 9223372036854775807  # positive bigint range
        batch_size = total_max_value / n
        for i in range(n):
            shard_id = shards[i][0]
            min_value = i * batch_size + 1
            max_value = (i+1) * batch_size
            # TODO: What happens if the "shard_max_size" is exceeded?
            # How does Citus handle the range paritioning?
            # To prevent creation of new shards in the first place, we may need
            # to further increase "shard_count" and/or "shard_max_size".
            cursor.execute('''
                UPDATE pg_dist_shard SET shardminvalue = %(min_value)s
                WHERE logicalrelid = %(table)s::regclass
                AND shardid = %(shard_id)s
            ''', {
                'table': '{schema}.{table}'.format(
                    schema=schema_name, table=table_name
                ),
                'min_value': min_value,
                'shard_id': shards[i][0]
            })
            cursor.execute('''
                UPDATE pg_dist_shard SET shardmaxvalue = %(max_value)s
                WHERE logicalrelid = %(table)s::regclass
                AND shardid = %(shard_id)s
            ''', {
                'table': '{schema}.{table}'.format(
                    schema=schema_name, table=table_name
                ),
                'max_value': max_value,
                'shard_id': shard_id
            })
            cursor.execute('''
                CREATE SEQUENCE {schema}.{table}_id_seq_{shard}
                MINVALUE %(min_value)s MAXVALUE %(max_value)s
            '''.format(schema=schema_name, table=table_name, shard=shard_id), {
                'min_value': min_value,
                'max_value': max_value
            })


def _create_schema_if_not_exists(connection, schema_name):
    cursor = connection.connection.cursor()
    cursor.execute('''
        SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname = %(schema)s);
    ''', {
        'schema': schema_name
    })
    schema = cursor.fetchone()
    if schema[0]:
        cursor.close()
        return True
    else:
        logger.debug('create schema "%s"', schema_name)
        sql = 'CREATE SCHEMA IF NOT EXISTS %s;' % schema_name
        cursor.execute(sql)
        cursor.close()
        return False


def _drop_schema(connection, schema_name):
    logger.debug('drop all table in schema "%s"', schema_name)
    # NOTE: The tables are dropped on the worker nodes, but the schemas
    # persist. This is not a problem, however.
    cursor = connection.connection.cursor()
    cursor.execute('DROP SCHEMA %s CASCADE;' % schema_name)
    cursor.close()


def _create_main_db_tables(connection):
    logger.debug(
        'create tables of models derived from %s for schema "public"',
        MainModel.__name__
    )
    MainModel.metadata.create_all(connection)


def _create_experiment_db_tables(connection, schema_name):
    logger.debug(
        'create tables of models derived from %s for schema "%s"',
        ExperimentModel.__name__, schema_name
    )
    # NOTE: We need to set the schema on copies of the tables otherwise
    # this messes up queries in a multi-tenancy use case.
    experiment_specific_metadata = sqlalchemy.MetaData(schema=schema_name)
    for name, table in ExperimentModel.metadata.tables.iteritems():
        table_copy = table.tometadata(experiment_specific_metadata)
    experiment_specific_metadata.create_all(connection)


# def _create_distributed_experiment_db_tables(connection, schema_name):
#     logger.debug(
#         'create distributed tables of models derived from %s for schema "%s"',
#         ExperimentModel.__name__, schema_name
#     )
#     experiment_specific_metadata = sqlalchemy.MetaData(schema=schema_name)
#     for name, table in ExperimentModel.metadata.tables.iteritems():
#         if table.is_distributed:
#             table_copy = table.tometadata(experiment_specific_metadata)
#     experiment_specific_metadata.create_all(connection)


@listens_for(sqlalchemy.pool.Pool, 'connect')
def _on_pool_connect(dbapi_con, connection_record):
    logger.debug(
        'database connection created for pool: %d',
        dbapi_con.get_backend_pid()
    )


@listens_for(sqlalchemy.pool.Pool, 'checkin')
def _on_pool_checkin(dbapi_con, connection_record):
    logger.debug(
        'database connection returned to pool: %d',
        dbapi_con.get_backend_pid()
    )


@listens_for(sqlalchemy.pool.Pool, 'checkout')
def _on_pool_checkout(dbapi_con, connection_record, connection_proxy):
    logger.debug(
        'database connection retrieved from pool: %d',
        dbapi_con.get_backend_pid()
    )


def create_db_session_factory():
    '''Creates a factory for creating a scoped database session that will use
    :class:`Query <tmlib.models.utils.Query>` to query the database.

    Returns
    -------
    sqlalchemy.orm.session.Session
    '''
    return sqlalchemy.orm.scoped_session(
        sqlalchemy.orm.sessionmaker(query_cls=Query)
    )


def delete_location(path):
    '''Deletes a location on disk.

    Parameters
    ----------
    path: str
        absolute path to directory or file
    '''
    if os.path.exists(path):
        logger.debug('remove location: %s', path)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)


def remove_location_upon_delete(cls):
    '''Decorator function for an database model class that
    automatically removes the `location` that represents an instance of the
    class on the filesystem once the corresponding row is deleted from the
    database table.

    Parameters
    ----------
    cls: tmlib.models.base.DeclarativeABCMeta
       implemenation of :class:`tmlib.models.base.FileSystemModel`

    Raises
    ------
    AttributeError
        when decorated class doesn't have a "location" attribute
    '''
    def after_delete_callback(mapper, connection, target):
        delete_location(target.location)

    sqlalchemy.event.listen(cls, 'after_delete', after_delete_callback)
    return cls


def exec_func_after_insert(func):
    '''Decorator function for a database model class that calls the
    decorated function after an `insert` event.

    Parameters
    ----------
    func: function

    Examples
    --------
    @exec_func_after_insert(lambda target: do_something())
    SomeClass(db.Model):

    '''
    def class_decorator(cls):
        def after_insert_callback(mapper, connection, target):
            func(mapper, connection, target)
        sqlalchemy.event.listen(cls, 'after_insert', after_insert_callback)
        return cls
    return class_decorator


class Query(sqlalchemy.orm.query.Query):

    '''A custom query class.'''

    def __init__(self, *args, **kwargs):
        super(Query, self).__init__(*args, **kwargs)

    def delete(self):
        '''Performs a bulk delete query.

        Returns
        -------
        int
            count of rows matched as returned by the database's "row count"
            feature

        Note
        ----
        Also removes locations of instances on the file system.
        '''
        classes = [d['type'] for d in self.column_descriptions]
        locations = list()
        for cls in classes:
            if cls.is_distributed:
                raise ValueError(
                    'Records of distributed model "%s" cannot be deleted '
                    'within a transaction.' % cls.__name__
                )
            if hasattr(cls, '_location'):
                locations.extend(self.from_self(cls._location).all())
            elif hasattr(cls, 'location'):
                instances = self.from_self(cls).all()
                locations.extend([(inst.location,) for inst in instances])
            if cls.__name__ == 'Experiment':
                raise ValueError(
                    'To delete an experiment delete the corresponding '
                    'reference object.'
                )
            elif cls.__name__ == 'ExperimentReference':
                experiments = self.from_self(cls.id).all()
                connection = self.session.get_bind()
                for exp in experiments:
                    logger.info('drop schema of experiment %d', exp.id)
                    schema = _SCHEMA_NAME_FORMAT_STRING.format(
                        experiment_id=exp.id
                    )
                    _drop_schema(connection, schema)
        # For performance reasons delete all rows via raw SQL without updating
        # the session and then enforce the session to update afterwards.
        logger.debug(
            'delete instances of class %s from database', cls.__name__
        )
        super(Query, self).delete(synchronize_session=False)
        self.session.expire_all()
        if locations:
            logger.debug('remove corresponding locations on disk')
            for loc in locations:
                if loc[0] is not None:
                    delete_location(loc[0])


class _SQLAlchemy_Session(object):

    '''A wrapper around an instance of an *SQLAlchemy* session
    that manages persistence of database model objects.

    An instance of this class will be exposed via
    :class:`MainSession <tmlib.models.utils.MainSession>` and
    :class:`ExperimentSession <tmlib.models.utils.ExperimentSession>`.

    Examples
    --------
    >>> import tmlib.models as tm
    >>> with tm.utils.MainSession() as session:
    >>>     print(session.drop_and_recreate(tm.Submission))

    '''

    def __init__(self, session, schema=None):
        '''
        Parameters
        ----------
        session: sqlalchemy.orm.session.Session
            *SQLAlchemy* database session
        schema: str, optional
            name of a database schema
        '''
        self._session = session
        self._schema = schema

    def __getattr__(self, attr):
        if hasattr(self._session, attr):
            return getattr(self._session, attr)
        elif hasattr(self, attr):
            return getattr(self, attr)
        else:
            raise AttributeError(
                'Object "%s" doens\'t have attribute "%s".'
                % (self.__class__.__name__, attr)
            )

    @property
    def connection(self):
        '''database connection'''
        return self._session.get_bind()

    def get_or_create(self, model, **kwargs):
        '''Gets an instance of a model class if it already exists or
        creates it otherwise.

        Parameters
        ----------
        model: type
            an implementation of :class:`tmlib.models.base.MainModel` or
            :class:`tmlib.models.base.ExperimentModel`
        **kwargs: dict
            keyword arguments for the instance that can be passed to the
            constructor of `model` or to
            :meth:`sqlalchemy.orm.query.query.filter_by`

        Returns
        -------
        tmlib.models.model
            an instance of `model`

        Note
        ----
        Adds and commits created instance. The approach can be useful when
        different processes may try to insert an instance constructed with the
        same arguments, but only one instance should be inserted and the other
        processes should re-use the instance without creation a duplication.
        The approach relies on uniqueness constraints of the corresponding table
        to decide whether a new entry would be considred a duplication.
        '''
        try:
            instance = self._session.query(model).\
                filter_by(**kwargs).\
                one()
            logger.debug('found existing instance: %r', instance)
        except sqlalchemy.orm.exc.NoResultFound:
            # We have to protect against situations when several worker
            # nodes are trying to insert the same row simultaneously.
            try:
                instance = model(**kwargs)
                self._session.add(instance)
                self._session.commit()
                logger.debug('created new instance: %r', instance)
            except sqlalchemy.exc.IntegrityError as err:
                logger.error(
                    'creation of %s instance failed:\n%s', model, str(err)
                )
                self._session.rollback()
                try:
                    instance = self._session.query(model).\
                        filter_by(**kwargs).\
                        one()
                    logger.debug('found existing instance: %r', instance)
                except:
                    raise
            except TypeError:
                raise TypeError(
                    'Wrong arugments for instantiation of model class "%s".'
                    % model.__name__
                )
            except:
                raise
        except:
            raise
        return instance

    def drop_and_recreate(self, model):
        '''Drops a database table and re-creates it. Also removes
        locations on disk for each row of the dropped table.

        Parameters
        ----------
        model: tmlib.models.MainModel or tmlib.models.ExperimentModel
            database model class

        Warning
        -------
        Disk locations are removed after the table is dropped. This can lead
        to inconsistencies between database and file system representation of
        `model` instances when the process is interrupted.
        '''
        connection = self._session.get_bind()
        locations_to_remove = []
        # We need to update the schema on each data model, such that tables
        # will be created for the correct experiment-specific schema and not
        # created for the "public" schema.
        experiment_specific_metadata = sqlalchemy.MetaData(schema=self._schema)
        for name, table in ExperimentModel.metadata.tables.iteritems():
            table_copy = table.tometadata(experiment_specific_metadata)
        table_name = '{schema}.{table}'.format(
            schema=self._schema, table=model.__table__.name
        )
        table = experiment_specific_metadata.tables[table_name]

        if table.exists(connection):
            if issubclass(model, FileSystemModel):
                model_instances = self._session.query(model).all()
                locations_to_remove = [m.location for m in model_instances]
            logger.info('drop table "%s"', table.name)
            self._session.commit()  # circumvent locking
            table.drop(connection)

        logger.info('create table "%s"', table.name)
        table.create(connection)
        logger.info('remove "%s" locations on disk', model.__name__)
        for loc in locations_to_remove:
            logger.debug('remove "%s"', loc)
            delete_location(loc)


class _Session(object):

    '''Class that provides access to all methods and attributes of
    :class:`sqlalchemy.orm.session.Session` and additional
    custom methods implemented in
    :class:`tmlib.models.utils._SQLAlchemy_Session`.

    Note
    ----
    The engine is cached and reused in case of a reconnection within the same
    Python process.

    Warning
    -------
    This is *not* thread-safe!
    '''

    def __init__(self, db_uri, schema=None):
        self._db_uri = db_uri
        self._schema = schema
        self._session_factory = create_db_session_factory()

    # @cached_property
    # def engine(self):
    #     '''sqlalchemy.engine: engine object for the currently used database'''
    #     return create_db_engine(self._db_uri)

    def __exit__(self, except_type, except_value, except_trace):
        if except_value:
            logger.debug('rollback session due to error')
            self._session.rollback()
        else:
            try:
                logger.debug('commit session')
                self._session.commit()
            except RuntimeError:
                logger.error('commit failed due to RuntimeError???')
        connection = self._session.get_bind()
        connection.close()
        self._session.close()


class MainSession(_Session):

    '''Session scopes for interaction with the main ``tissuemaps`` database.
    All changes get automatically committed at the end of the interaction.
    In case of an error, a rollback is issued.

    Examples
    --------
    >>> import tmlib.models as tm
    >>> with tm.utils.MainSession() as session:
    >>>    print(session.query(tm.ExperimentReference).all())

    See also
    --------
    :class:`tmlib.models.base.MainModel`
    '''

    def __init__(self, db_uri=None):
        '''
        Parameters
        ----------
        db_uri: str, optional
            URI of the ``tissuemaps`` database; defaults to the value of
            :attr:`db_uri <tmlib.config.DefaultConfig.db_uri>`
        '''
        if db_uri is None:
            db_uri = cfg.db_master_uri
        super(MainSession, self).__init__(db_uri)
        self._schema = None
        self._engine = create_db_engine(db_uri)
        _assert_db_exists(self._engine)

    def __enter__(self):
        connection = self._engine.connect()
        self._session_factory.configure(bind=connection)
        self._session = _SQLAlchemy_Session(
            self._session_factory(), self._schema
        )
        return self._session


class ExperimentSession(_Session):

    '''Session scopes for interaction with an experiment-secific database.
    All changes get automatically committed at the end of the interaction.
    In case of an error, a rollback is issued.

    Examples
    --------
    >>> import tmlib.models as tm
    >>> with tm.utils.ExperimentSession(experiment_id=1) as session:
    >>>     print(session.query(tm.Plate).all())

    See also
    --------
    :class:`tmlib.models.base.ExperimentModel`
    '''

    def __init__(self, experiment_id, db_uri=None):
        '''
        Parameters
        ----------
        experiment_id: int
            ID of the experiment that should be queried
        db_uri: str, optional
            URI of the ``tissuemaps`` database; defaults to the value of
            :attr:`db_uri <tmlib.config.LibraryConfig.db_uri>`
        '''
        if db_uri is None:
            db_uri = cfg.db_master_uri
        self.experiment_id = experiment_id
        logger.debug('create session for experiment %d', self.experiment_id)
        self._engine = create_db_engine(db_uri)
        schema = _SCHEMA_NAME_FORMAT_STRING.format(
            experiment_id=self.experiment_id
        )
        logger.debug('schema: "%s"', schema)
        super(ExperimentSession, self).__init__(db_uri, schema)

    def __enter__(self):
        connection = self._engine.connect()
        exists = _create_schema_if_not_exists(connection, self._schema)
        if not exists:
            _create_experiment_db_tables(connection, self._schema)
            _customize_distributed_tables(connection, self._schema)
        _set_search_path(connection, self._schema)
        self._session_factory.configure(bind=connection)
        self._session = _SQLAlchemy_Session(
            self._session_factory(), self._schema
        )
        return self._session


class _Connection(object):

    '''A "raw" database connection which uses autocommit mode and is not
    part of a transaction.

    Such connections are required to issues statements such as
    ``CREATE DATABASE``, for example.

    Warning
    -------
    Only use a raw connection when absolutely required and when you know what
    you are doing.
    '''

    def __init__(self, db_uri):
        '''
        Parameters
        ----------
        db_uri: str
            URI of the database to connect to in the format required by
            *SQLAlchemy*
        '''
        self._db_uri = db_uri
        self._engine = create_db_engine(self._db_uri)

    def __enter__(self):
        # NOTE: We need to run queries outside of a transaction in Postgres
        # autocommit mode.
        self._connection = self._engine.raw_connection()
        self._connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        self._cursor = self._connection.cursor(cursor_factory=NamedTupleCursor)
        # NOTE: To achieve high throughput on UPDATE or DELETE, we
        # need to perform queries in parallel under the assumption that
        # order of records is not important (i.e. that they are commutative).
        # https://docs.citusdata.com/en/v6.0/performance/scaling_data_ingestion.html#real-time-updates-0-50k-s
        self._cursor.execute('''
            SET citus.shard_replication_factor = 1;
            SET citus.all_modifications_commutative TO on;
        ''')
        return self

    def __exit__(self, except_type, except_value, except_trace):
        # NOTE: The connection is not actually closed, but rather returned to
        # the pool.
        self._cursor.close()
        self._connection.close()

    def __getattr__(self, attr):
        if hasattr(self._cursor, attr):
            return getattr(self._cursor, attr)
        elif hasattr(self, attr):
            return getattr(self, attr)
        else:
            raise AttributeError(
                'Object "%s" doens\'t have attribute "%s".'
                % (self.__class__.__name__, attr)
            )


class ExperimentConnection(_Connection):

    '''Database connection for executing raw SQL statements for an
    experiment-specific database outside of a transaction context.

    Examples
    --------
    >>> import tmlib.models as tm
    >>> with tm.utils.ExperimentConnection(experiment_id=1) as connection:
    >>>     connection.execute('SELECT mapobject_id, value FROM feature_values;')
    >>>     print(connection.fetchall())

    Warning
    -------
    Use raw connections only if absolutely necessary, such as for inserting
    into or updating distributed tables. Otherwise use
    :class:`ExperimentSession <tmlib.models.utils.ExperimentSession>`.

    See also
    --------
    :class:`tmlib.models.base.ExperimentModel`
    '''

    def __init__(self, experiment_id):
        '''
        Parameters
        ----------
        experiment_id: int
            ID of the experiment that should be queried
        '''
        super(ExperimentConnection, self).__init__(cfg.db_master_uri)
        self._schema = _SCHEMA_NAME_FORMAT_STRING.format(
            experiment_id=experiment_id
        )
        self.experiment_id = experiment_id
        self._shard_lut = dict()

    def __enter__(self):
        # NOTE: We need to run queries outside of a transaction in Postgres
        # autocommit mode.
        self._connection = self._engine.raw_connection()
        self._connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        exists = _create_schema_if_not_exists(self._connection, self._schema)
        if not exists:
            _create_experiment_db_tables(self._connection, self._schema)
            _customize_distributed_tables(connection, self._schema)
        _set_search_path(self._connection, self._schema)
        self._cursor = self._connection.cursor(cursor_factory=NamedTupleCursor)
        # NOTE: To achieve high throughput on UPDATE or DELETE, we
        # need to perform queries in parallel under the assumption that
        # order of records is not important (i.e. that they are commutative).
        # https://docs.citusdata.com/en/v6.0/performance/scaling_data_ingestion.html#real-time-updates-0-50k-s
        logger.debug('make modifications commutative')
        self._cursor.execute('''
            SET citus.shard_replication_factor = 1;
            SET citus.all_modifications_commutative TO on;
        ''')
        return self

    def get_unique_id(self, model_class):
        '''Gets a unique value for the distribution column.

        Parameters
        ----------
        model_class: str
            class dervired from
            :class:`ExperimentModel <tmlib.models.base.ExperimentModel>`
            for which a shard should be selected
        '''
        self._cursor.execute('''
            SELECT nextval FROM nextval(%(sequence)s);
        ''', {
            'sequence': '{table}_id_seq'.format(
                table=model_class.__table__.name
            )
        })
        return self._cursor.fetchone()[0]

    def get_shard_id(self, model_class):
        '''Selects a single shard at random from all available shards.
        The ID of the selected shard gets cached, such that subsequent calls
        will return the same identifier.

        Parameters
        ----------
        model_class: str
            class dervired from
            :class:`ExperimentModel <tmlib.models.base.ExperimentModel>`
            for which a shard should be selected

        Returns
        -------
        int
            ID of the selected shard

        Raise
        -----
        ValueError
            when the `model_class` does not represent a distributed table
        '''
        if not model_class.is_distributed:
            raise ValueError(
                'Shard selection not possible, since provided model class does '
                'not represent a distributed database table.'
            )
        if model_class.__name__ in self._shard_lut:
            return self._shard_lut[model_class.__name__]
        self._cursor.execute('''
            SELECT shardid FROM pg_dist_shard
            WHERE logicalrelid = %(table)s::regclass
            ORDER BY random()
            LIMIT 1
        ''', {
            'table': model_class.__table__.name,
        })
        shard_id = self._cursor.fetchone()[0]
        self._shard_lut[model_class.__name__] = shard_id
        return shard_id

    def get_shard_specific_unique_id(self, model_class, shard_id):
        '''Gets a unique, but shard-specific value for the distribution column.

        Parameters
        ----------
        model_class: str
            class dervired from
            :class:`ExperimentModel <tmlib.models.base.ExperimentModel>`
            for which a shard should be selected
        shard_id: int
            ID of a shard that is located on the worker server to which the
            connection was established
        '''
        self._cursor.execute('''
            SELECT nextval FROM nextval(%(sequence)s);
        ''', {
            'sequence': '{table}_id_seq_{shard}'.format(
                table=model_class.__table__.name, shard=shard_id
            )
        })
        return self._cursor.fetchone()[0]


class MainConnection(_Connection):

    '''Database connection for executing raw SQL statements for the
    main ``tissuemaps`` database outside of a transaction context.

    Examples
    --------
    >>> import tmlib.models as tm
    >>> with tm.utils.MainConnection() as connection:
    >>>     connection.execute('SELECT name FROM plates;')
    >>>     print(connection.fetchall())

    Warning
    -------
    Use raw connnections only if absolutely necessary, such as when inserting
    into or updating distributed tables. Otherwise use
    :class:`MainSession <tmlib.models.utils.MainSession>`.

    See also
    --------
    :class:`tmlib.models.base.MainModel`
    '''

    def __init__(self):
        super(MainConnection, self).__init__(cfg.db_master_uri)


class ExperimentWorkerConnection(_Connection):

    '''Database connection for executing raw SQL statements on a database
    "worker" server to target individual shards of a distributed,
    experiment-specific table. A random server will be chosen automatically
    for load balancing.

    Warning
    -------
    Use raw connections only if absolutely necessary, such as for inserting
    into or updating distributed tables. Otherwise use
    :class:`ExperimentSession <tmlib.models.utils.ExperimentSession>`.

    See also
    --------
    :class:`tmlib.models.base.ExperimentModel`
    '''

    def __init__(self, experiment_id):
        '''
        Parameters
        ----------
        experiment_id: int
            ID of the experiment that should be queried
        '''
        with MainConnection() as connection:
            connection.execute('''
                SELECT nodename, nodeport FROM pg_dist_shard_placement
                ORDER BY random()
                LIMIT 1
            ''')
            host, port = connection.fetchone()
        db_uri = cfg.build_db_worker_uri(host, port)
        super(ExperimentWorkerConnection, self).__init__(db_uri)
        self._schema = _SCHEMA_NAME_FORMAT_STRING.format(
            experiment_id=experiment_id
        )
        self._host = host
        self._port = port
        self._shard_lut = dict()
        self.experiment_id = experiment_id

    def get_shard_id(self, model_class):
        '''Selects a single shard at random from all available shards on the
        worker server to which the connection was established. The ID of the
        selected shard gets cached, such that subsequent calls will return
        the same identifier.

        Parameters
        ----------
        model_class: str
            class dervired from
            :class:`ExperimentModel <tmlib.models.base.ExperimentModel>`
            for which a shard should be selected

        Returns
        -------
        int
            ID of the selected shard

        Raise
        -----
        ValueError
            when the `model_class` does not represent a distributed table
        '''
        if not model_class.is_distributed:
            raise ValueError(
                'Shard selection not possible, since provided model class does '
                'not represent a distributed database table.'
            )
        if model_class.__name__ in self._shard_lut:
            return self._shard_lut[model_class.__name__]
        with MainConnection() as connection:
            connection.execute('''
                SELECT s.shardid FROM pg_dist_shard AS s
                JOIN pg_dist_shard_placement AS p ON s.shardid = p.shardid
                WHERE s.logicalrelid = %(table)s::regclass
                AND p.nodename = %(host)s
                AND p.nodeport = %(port)s
                ORDER BY random()
                LIMIT 1
            ''', {
                'table': model_class.__table__.name,
                'host': self._host,
                'port': self._port
            })
            shard_id = connection.fetchone()[0]
            self._shard_lut[model_class.__name__] = shard_id
            return shard_id

    def get_shard_specific_unique_id(self, model_class, shard_id):
        '''Gets a unique, but shard-specific value for the distribution column.

        Parameters
        ----------
        model_class: str
            class dervired from
            :class:`ExperimentModel <tmlib.models.base.ExperimentModel>`
            for which a shard should be selected
        shard_id: int
            ID of a shard that is located on the worker server to which the
            connection was established
        '''
        with MainConnection() as connection:
            connection.execute('''
                SELECT nextval FROM nextval(%(sequence)s);
            ''', {
                'sequence': '{table}_id_seq_{shard}'.format(
                    table=model_class.__table__.name, shard=shard_id
                )
            })
            return connection.fetchone()[0]

    def __enter__(self):
        self._connection = self._engine.raw_connection()
        # self._connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        _set_search_path(self._connection, self._schema)
        self._cursor = self._connection.cursor(cursor_factory=NamedTupleCursor)
        # NOTE: To achieve high throughput on UPDATE or DELETE, we
        # need to perform queries in parallel under the assumption that
        # order of records is not important (i.e. that they are commutative).
        # https://docs.citusdata.com/en/v6.0/performance/scaling_data_ingestion.html#real-time-updates-0-50k-s
        logger.debug('make modifications commutative')
        self._cursor.execute('''
            SET citus.shard_replication_factor = 1;
            SET citus.all_modifications_commutative TO on;
        ''')
        return self
