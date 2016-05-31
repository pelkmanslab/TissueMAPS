import os
import yaml
import glob
import time
import logging
import numpy as np
import datetime
from natsort import natsorted
from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
import gc3libs
from gc3libs.quantity import Duration
from gc3libs.quantity import Memory

import tmlib.models as tm
from tmlib import utils
from tmlib.readers import JsonReader
from tmlib.workflow import BgEngine
from tmlib.writers import JsonWriter
from tmlib.errors import JobDescriptionError
from tmlib.errors import WorkflowError
from tmlib.errors import WorkflowDescriptionError
from tmlib.workflow.jobs import RunJob
from tmlib.workflow.jobs import SingleRunJobCollection
from tmlib.workflow.jobs import CollectJob
from tmlib.workflow.workflow import WorkflowStep

logger = logging.getLogger(__name__)


class BasicClusterRoutines(object):

    '''Abstract base class for submission of jobs to a cluster.'''

    __metaclass__ = ABCMeta

    @property
    def datetimestamp(self):
        '''
        Returns
        -------
        str
            datetime stamp in the form "year-month-day_hour:minute:second"
        '''
        return utils.create_datetimestamp()

    @property
    def timestamp(self):
        '''
        Returns
        -------
        str
            time stamp in the form "hour:minute:second"
        '''
        return utils.create_timestamp()


class ClusterRoutines(BasicClusterRoutines):

    '''Abstract base class for API classes, which provide methods for
    for large scale image processing on a batch cluster.

    Each workflow step must implement this class and decorate it with
    :py:function:`tmlib.workflow.api.api` to register it for use in
    command line interface and worklow.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, experiment_id, verbosity):
        '''
        Parameters
        ----------
        experiment_id: int
            ID of the processed experiment
        verbosity: int
            logging level

        Attributes
        ----------
        experiment_id: int
            ID of the processed experiment
        verbosity: int
            logging level
        workflow_location: str
            absolute path to location where workflow related data should be
            stored
        '''
        super(ClusterRoutines, self).__init__()
        self.experiment_id = experiment_id
        self.verbosity = verbosity
        with tm.utils.Session() as session:
            experiment = session.query(tm.Experiment).\
                get(self.experiment_id)
            self.workflow_location = experiment.workflow_location

    @property
    def step_name(self):
        '''str: name of the step'''
        return self.__module__.split('.')[-2]

    @staticmethod
    def _create_batches(li, n):
        # Create a list of lists from a list, where each sublist has length n
        n = max(1, n)
        return [li[i:i + n] for i in range(0, len(li), n)]

    @utils.autocreate_directory_property
    def step_location(self):
        '''str: location were step-specific data is stored'''
        return os.path.join(self.workflow_location, self.step_name)

    @utils.autocreate_directory_property
    def log_location(self):
        '''str: location where log files are stored'''
        return os.path.join(self.step_location, 'log')

    @utils.autocreate_directory_property
    def batches_location(self):
        '''str: location where job description files are stored'''
        return os.path.join(self.step_location, 'batches')

    def get_batches_from_files(self):
        '''Gets batches from files and combine them into
        the format required by the `create_jobs()` method.

        Returns
        -------
        dict
            job descriptions

        Raises
        ------
        :py:exc:`tmlib.errors.JobDescriptionError`
            when no job descriptor files are found
        '''
        batches = dict()
        batches['run'] = list()
        run_job_files = glob.glob(
            os.path.join(self.batches_location, '*_run_*.batch.json')
        )
        if not run_job_files:
            raise JobDescriptionError('No batch files found.')
        collect_job_files = glob.glob(
            os.path.join(self.batches_location, '*_collect.batch.json')
        )

        for f in run_job_files:
            batch = self.read_batch_file(f)
            batches['run'].append(batch)
        if collect_job_files:
            f = collect_job_files[0]
            batches['collect'] = self.read_batch_file(f)

        return batches

    def get_log_output_from_files(self, job_id):
        '''Gets log outputs (standard output and error) from files.

        Parameters
        ----------
        job_id: int
            one-based job identifier number

        Returns
        -------
        Dict[str, str]
            "stdout" and "stderr" for the given job

        Note
        ----
        In case there are several log files present for the given the most
        recent one will be used (sorted by submission date and time point).
        '''
        if job_id is not None:
            stdout_files = glob.glob(
                os.path.join(self.log_location, '*_run*_%.6d*.out' % job_id)
            )
            stderr_files = glob.glob(
                os.path.join(self.log_location, '*_run*_%.6d*.err' % job_id)
            )
            if not stdout_files or not stderr_files:
                raise IOError('No log files found for run job # %d' % job_id)
        else:
            stdout_files = glob.glob(
                os.path.join(self.log_location, '*_collect*.out')
            )
            stderr_files = glob.glob(
                os.path.join(self.log_location, '*_collect_*.err')
            )
            if not stdout_files or not stderr_files:
                raise IOError('No log files found for collect job')
        # Take the most recent log files
        log = dict()
        with open(natsorted(stdout_files)[-1], 'r') as f:
            log['stdout'] = f.read()
        with open(natsorted(stderr_files)[-1], 'r') as f:
            log['stderr'] = f.read()
        return log

    def list_output_files(self, batches):
        '''Lists all output files that should be created by the step.

        Parameters
        ----------
        batches: List[dict]
            job descriptions
        '''
        files = list()
        if batches['run']:
            run_files = utils.flatten([
                self._make_paths_absolute(j)['outputs'].values()
                for j in batches['run']
            ])
            if all([isinstance(f, list) for f in run_files]):
                run_files = utils.flatten(run_files)
                if all([isinstance(f, list) for f in run_files]):
                    run_files = utils.flatten(run_files)
                files.extend(run_files)
            else:
                files.extend(run_files)
        if 'collect' in batches.keys():
            outputs = batches['collect']['outputs']
            collect_files = utils.flatten(outputs.values())
            if all([isinstance(f, list) for f in collect_files]):
                collect_files = utils.flatten(collect_files)
                if all([isinstance(f, list) for f in collect_files]):
                    collect_files = utils.flatten(collect_files)
                files.extend(collect_files)
            else:
                files.extend(collect_files)
        return files

    def list_input_files(self, batches):
        '''Provides a list of all input files that are required by the step.

        Parameters
        ----------
        batches: List[dict]
            job descriptions
        '''
        files = list()
        if batches['run']:
            run_files = utils.flatten([
                self._make_paths_absolute(j)['inputs'].values()
                for j in batches['run']
            ])
            if all([isinstance(f, list) for f in run_files]):
                run_files = utils.flatten(run_files)
                if all([isinstance(f, list) for f in run_files]):
                    run_files = utils.flatten(run_files)
                files.extend(run_files)
            elif any([isinstance(f, dict) for f in run_files]):
                files.extend(utils.flatten([
                    utils.flatten(f.values())
                    for f in run_files if isinstance(f, dict)
                ]))
            else:
                files.extend(run_files)
        return files

    def build_batch_filename_for_run_job(self, job_id):
        '''Builds the name of a batch file for a run job.

        Parameters
        ----------
        job_id: int
            one-based job identifier number

        Returns
        -------
        str
            absolute path to the file that holds the description of the
            job with the given `job_id`

        Note
        ----
        The total number of jobs is limited to 10^6.
        '''
        return os.path.join(
            self.batches_location,
            '%s_run_%.6d.batch.json' % (self.step_name, job_id)
        )

    def build_batch_filename_for_collect_job(self):
        '''Builds the name of a batch file for a collect job.

        Returns
        -------
        str
            absolute path to the file that holds the description of the
            job with the given `job_id`
        '''
        return os.path.join(
            self.batches_location,
            '%s_collect.batch.json' % self.step_name
        )

    def _make_paths_absolute(self, batch):
        for key, value in batch['inputs'].items():
            if isinstance(value, dict):
                for k, v in batch['inputs'][key].items():
                    if isinstance(v, list):
                        batch['inputs'][key][k] = [
                            os.path.join(self.workflow_location, sub_v)
                            for sub_v in v
                        ]
                    else:
                        batch['inputs'][key][k] = os.path.join(
                            self.workflow_location, v
                        )
            elif isinstance(value, list):
                if len(value) == 0:
                    continue
                if isinstance(value[0], list):
                    for i, v in enumerate(value):
                        batch['inputs'][key][i] = [
                            os.path.join(self.workflow_location, sub_v)
                            for sub_v in v
                        ]
                else:
                    batch['inputs'][key] = [
                        os.path.join(self.workflow_location, v)
                        for v in value
                    ]
            else:
                raise TypeError(
                    'Value of "inputs" must have type list or dict.'
                )
        for key, value in batch['outputs'].items():
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                if isinstance(value[0], list):
                    for i, v in enumerate(value):
                        batch['outputs'][key][i] = [
                            os.path.join(self.workflow_location, sub_v)
                            for sub_v in v
                        ]
                else:
                    batch['outputs'][key] = [
                        os.path.join(self.workflow_location, v)
                        for v in value
                    ]
            elif isinstance(value, basestring):
                batch['outputs'][key] = os.path.join(
                    self.workflow_location, value
                )
            else:
                raise TypeError(
                    'Value of "outputs" must have type list or str.'
                )
        return batch

    def read_batch_file(self, filename):
        '''Read batches from JSON file.

        Parameters
        ----------
        filename: str
            absolute path to the *.job* file that contains the description
            of a single job

        Returns
        -------
        dict
            batch

        Raises
        ------
        tmlib.errors.WorkflowError
            when `filename` does not exist

        Note
        ----
        The relative paths for "inputs" and "outputs" are made absolute.
        '''
        if not os.path.exists(filename):
            raise WorkflowError(
                'Job description file does not exist: %s.\n'
                'Initialize the step first by calling the "init" method.'
                % filename
            )
        with JsonReader(filename) as f:
            batch = f.read()
            return self._make_paths_absolute(batch)

    @staticmethod
    def _check_io_description(batches):
        if not all([
                isinstance(batch['inputs'], dict)
                for batch in batches['run']]):
            raise TypeError('"inputs" must have type dictionary')
        if not all([
                isinstance(batch['inputs'].values(), list)
                for batch in batches['run']]):
            raise TypeError('Elements of "inputs" must have type list')
        if not all([
                isinstance(batch['outputs'], dict)
                for batch in batches['run']]):
            raise TypeError('"outputs" must have type dictionary')
        if not all([
                all([isinstance(o, list) for o in batch['outputs'].values()])
                for batch in batches['run']]):
            raise TypeError('Elements of "outputs" must have type list.')
        if 'collect' in batches:
            batch = batches['collect']
            if not isinstance(batch['inputs'], dict):
                raise TypeError('"inputs" must have type dictionary')
            if not isinstance(batch['inputs'].values(), list):
                raise TypeError('Elements of "inputs" must have type list')
            if not isinstance(batch['outputs'], dict):
                raise TypeError('"outputs" must have type dictionary')
            if not all([isinstance(o, list) for o in batch['outputs'].values()]):
                raise TypeError('Elements of "outputs" must have type list')

    def _make_paths_relative(self, batch):
        for key, value in batch['inputs'].items():
            if isinstance(value, dict):
                for k, v in batch['inputs'][key].items():
                    if isinstance(v, list):
                        batch['inputs'][key][k] = [
                            os.path.relpath(sub_v, self.workflow_location)
                            for sub_v in v
                        ]
                    else:
                        batch['inputs'][key][k] = os.path.relpath(
                            v, self.workflow_location
                        )
            elif isinstance(value, list):
                if len(value) == 0:
                    continue
                if isinstance(value[0], list):
                    for i, v in enumerate(value):
                        batch['inputs'][key][i] = [
                            os.path.relpath(sub_v, self.workflow_location)
                            for sub_v in v
                        ]
                else:
                    batch['inputs'][key] = [
                        os.path.relpath(v, self.workflow_location)
                        for v in value
                    ]
            else:
                raise TypeError(
                    'Value of "inputs" must have type list or dict.'
                )
        for key, value in batch['outputs'].items():
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                if isinstance(value[0], list):
                    for i, v in enumerate(value):
                        batch['outputs'][key][i] = [
                            os.path.relpath(sub_v, self.workflow_location)
                            for sub_v in v
                        ]
                else:
                    batch['outputs'][key] = [
                        os.path.relpath(v, self.workflow_location)
                        for v in value
                    ]
            elif isinstance(value, basestring):
                batch['outputs'][key] = os.path.relpath(
                    value, self.workflow_location
                )
            else:
                raise TypeError(
                    'Value of "outputs" must have type list or str.'
                )
        return batch

    def write_batch_files(self, batches):
        '''Write batches to files as JSON.

        Parameters
        ----------
        batches: List[dict]
            job descriptions

        Note
        ----
        The paths for "inputs" and "outputs" are made relative to the
        experiment directory.
        '''
        self._check_io_description(batches)
        for batch in batches['run']:
            logger.debug('make paths relative to experiment directory')
            batch = self._make_paths_relative(batch)
            batch_file = self.build_batch_filename_for_run_job(batch['id'])
            with JsonWriter(batch_file) as f:
                f.write(batch)
        if 'collect' in batches.keys():
            batch = self._make_paths_relative(batches['collect'])
            batch_file = self.build_batch_filename_for_collect_job()
            with JsonWriter(batch_file) as f:
                f.write(batch)

    def _build_run_command(self, job_id):
        command = [self.step_name]
        command.extend(['-v' for x in xrange(self.verbosity)])
        command.append(self.experiment_id)
        command.extend(['run', '--job', str(job_id)])
        return command

    def _build_collect_command(self):
        command = [self.step_name]
        command.extend(['-v' for x in xrange(self.verbosity)])
        command.append(self.experiment_id)
        command.extend(['collect'])
        return command

    @abstractmethod
    def run_job(self, batch):
        '''Runs an individual job.

        Parameters
        ----------
        batch: dict
            description of the job
        '''
        pass

    @abstractmethod
    def delete_previous_job_output(self):
        '''Deletes the output of a previous submission.
        '''
        pass

    @abstractmethod
    def collect_job_output(self, batch):
        '''Collects the output of jobs and fuse them if necessary.

        Parameters
        ----------
        batches: List[dict]
            job descriptions
        **kwargs: dict
            additional variable input arguments as key-value pairs
        '''
        pass

    @abstractmethod
    def create_batches(self, args):
        '''Creates job descriptions with information required for the creation
        and processing of individual jobs.

        Parameters
        ----------
        args: tmlib.args.Args
            an instance of an implemented subclass of the `Args` base class

        There are two phases:
            * *run* phase: collection of tasks that are processed in parallel
            * *collect* phase: a single task that is processed once the
              *run* phase is terminated successfully

        Each batch (element of the *run* batches) must provide the
        following key-value pairs:
            * "id": one-based job identifier number (*int*)
            * "inputs": absolute paths to input files required to run the job
              (Dict[*str*, List[*str*]])
            * "outputs": absolute paths to output files produced the job
              (Dict[*str*, List[*str*]])

        In case a *collect* job is required, the corresponding batch must
        provide the following key-value pairs:
            * "inputs": absolute paths to input files required to collect job
              output of the *run* phase (Dict[*str*, List[*str*]])
            * "outputs": absolute paths to output files produced by the job
              (Dict[*str*, List[*str*]])

        A *collect* job description can have the optional key "removals", which
        provides a list of strings indicating which of the inputs are removed
        during the *collect* phase.

        A complete batches has the following structure::

            {
                "run": [
                    {
                        "id": ,            # int
                        "inputs": ,        # list or dict,
                        "outputs": ,       # list or dict,
                    },
                    ...
                ]
                "collect":
                    {
                        "inputs": ,        # list or dict,
                        "outputs": ,       # list or dict
                    }
            }

        Returns
        -------
        Dict[str, List[dict] or dict]
            job descriptions
        '''
        pass

    def print_job_descriptions(self, batches):
        '''Prints `batches` to standard output in YAML format.

        Parameters
        ----------
        batches: Dict[List[dict]]
            description of inputs and outputs or individual jobs
        '''
        print yaml.safe_dump(batches, default_flow_style=False)

    def create_step(self, submission_id):
        '''Creates the workflow step.

        Parameters
        ----------
        submission_id: int
            ID of the corresponding submission

        Returns
        -------
        tmlib.workflow.WorkflowStep
        '''
        logger.debug('create workflow step for submission %d', submission_id)
        return WorkflowStep(
            name=self.step_name,
            submission_id=submission_id
        )

    def create_run_jobs(self, submission_id, job_ids, duration, memory, cores):
        '''Creates jobs for the parallel "run" phase of the step.

        Parameters
        ----------
        submission_id: int
            ID of the corresponding submission
        job_ids: int
            IDs of jobs that should be created
        duration: str, optional
            computational time that should be allocated for a single job;
            in HH:MM:SS format (default: ``None``)
        memory: int, optional
            amount of memory in Megabyte that should be allocated for a single
            job (default: ``None``)
        cores: int, optional
            number of CPU cores that should be allocated for a single job
            (default: ``None``)

        Returns
        -------
        tmlib.workflow.jobs.SingleRunJobCollection
            run jobs
        '''
        logger.info('create run jobs for submission %d', submission_id)
        logger.debug('allocated time for run jobs: %s', duration)
        logger.debug('allocated memory for run jobs: %d MB', memory)
        logger.debug('allocated cores for run jobs: %d', cores)

        run_jobs = SingleRunJobCollection(
            step_name=self.step_name,
            submission_id=submission_id
        )
        for j in job_ids:
            job = RunJob(
                step_name=self.step_name,
                arguments=self._build_run_command(job_id=j),
                output_dir=self.log_location,
                job_id=j,
                submission_id=submission_id
            )
            if duration:
                job.requested_walltime = Duration(duration)
            if memory:
                job.requested_memory = Memory(memory, Memory.MB)
            if cores:
                if not isinstance(cores, int):
                    raise TypeError(
                        'Argument "cores" must have type int.'
                    )
                if not cores > 0:
                    raise ValueError(
                        'The value of "cores" must be positive.'
                    )
                job.requested_cores = cores
            run_jobs.add(job)
        return run_jobs

    def create_collect_job(self, submission_id):
        '''Creates job for the "collect" phase of the step.

        Parameters
        ----------
        submission_id: int
            ID of the corresponding submission

        Returns
        -------
        tmlib.workflow.jobs.CollectJob
            collect job

        Note
        ----
        Duration defaults to 2 hours and memory to 3800 megabytes.
        '''
        logger.info('create collect job for submission %d', submission_id)
        duration = Duration('02:00:00')
        memory = Memory(3800, Memory.MB)
        cores = 1
        logger.debug('allocated time for collect job: %s', duration)
        logger.debug('allocated memory for collect job: %d MB', memory)
        logger.debug('allocated cores for collect job: %d', cores)
        collect_job = CollectJob(
            step_name=self.step_name,
            arguments=self._build_collect_command(),
            output_dir=self.log_location,
            submission_id=submission_id
        )
        collect_job.requested_walltime = duration
        collect_job.requested_memory = memory
        collect_job.requested_cores = cores
        return collect_job

