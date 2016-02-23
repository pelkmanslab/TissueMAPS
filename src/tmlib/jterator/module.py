import os
import sys
import re
import logging
# import imp
import importlib
import traceback
import collections
import numpy as np
import rpy2.robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from cStringIO import StringIO
from . import path_utils
from ..errors import PipelineRunError

logger = logging.getLogger(__name__)


class CaptureOutput(dict):
    '''
    Class for capturing standard output and error of function calls
    and redirecting the STDOUT and STDERR strings to a dictionary.

    Examples
    --------
    with CaptureOutput() as output:
        my_function(arg)
    '''
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self._stringio_out = StringIO()
        sys.stderr = self._stringio_err = StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        output = self._stringio_out.getvalue()
        error = self._stringio_err.getvalue()
        self.update({'stdout': output, 'stderr': error})


class ImageProcessingModule(object):
    '''
    Class for a Jterator module, the building block of a Jterator pipeline.
    '''

    def __init__(self, name, module_file, handles_description):
        '''
        Initiate Module class.

        Parameters
        ----------
        name: str
            name of the module
        module_file: str
            path to program file that should be executed
        handles_description: dict
            description of module input/output

        Returns
        -------
        tmlib.jterator.module.ImageProcessingModule
        '''
        self.name = name
        self.module_file = module_file
        self.handles_description = handles_description
        self.outputs = dict()

    def build_log_filenames(self, log_dir, job_id):
        '''
        Build names of log-files into which the module will write
        standard output and error of the current job.

        Parameters
        ----------
        log_dir: str
            path to directory for log output
        job_id: int
            one-based job index

        Returns
        -------
        Dict[str, str]
            absolute path to files for standard output and error
        '''
        out_file = os.path.join(log_dir, '%s_%.5d.out' % (self.name, job_id))
        err_file = os.path.join(log_dir, '%s_%.5d.err' % (self.name, job_id))
        return {'stdout': out_file, 'stderr': err_file}

    def build_figure_filename(self, figures_dir, job_id):
        '''
        Build name of figure file into which module will write figure output
        of the current job.

        Parameters
        ----------
        figures_dir: str
            path to directory for figure output
        job_id: int
            one-based job index

        Returns
        -------
        str
            absolute path to the figure file
        '''
        figure_file = os.path.join(figures_dir, '%s_%.5d.html'
                                   % (self.name, job_id))
        return figure_file

    def build_error_message(self, input_data, stdout, stderr, message=''):
        '''
        Build a custom error massage that provides the standard input as well
        as standard output and error for an executed module.

        Parameters
        ----------
        input_data: dict
            the data parsed as input arguments to the module
        stdout: str
            standard output of the module execution
        stderr: str
            standard error of the module execution

        Returns
        -------
        str
            error message
        '''
        message = '\n\n\nExecution of module "{0}" failed:\n'.format(self.name)                                                                    
        if input_data:
            message += '\n' + '---[ Module input arguments ]---' \
                .ljust(80, '-') + '\n'
            for key, value in input_data.iteritems():
                message += '"{k}":\n{v}\n\n'.format(k=key, v=value)
        message += '\n' + '---[ Module standard output ]---' \
            .ljust(80, '-') + '\n' + stdout
        message += '\n' + '---[ Module standard error ]---' \
            .ljust(80, '-') + '\n' + stderr
        self.error_message = message
        return self.error_message

    def write_output_and_errors(self, stdout_file, stdout_data,
                                stderr_file, stderr_data):
        '''
        Write standard output and error to log file.

        Parameters
        ----------
        stdout_data: str
            standard output of the module execution
        stderr_data: str
            standard error of the module execution
        '''
        with open(stdout_file, 'w+') as output_log:
            output_log.write(stdout_data)
        with open(stderr_file, 'w+') as error_log:
            error_log.write(stderr_data)

    @property
    def language(self):
        '''
        Returns
        -------
        str
            language of the module (e.g. "python")
        '''
        return path_utils.determine_language(self.module_file)

    def _exec_m_module(self, inputs, output_names, engine):
        logger.debug('adding module to Matlab path: "%s"' % self.module_file)
        # engine.eval('addpath(\'{0}\');'.format(os.path.dirname(self.module_file)))
        module_name = os.path.splitext(os.path.basename(self.module_file))[0]
        engine.eval('import \'jtlib.modules.{0}\''.format(module_name))
        logger.debug('evaluating Matlab function with INPUTS: "%s"',
                     '", "'.join(inputs.keys()))
        for name, value in inputs.iteritems():
            engine.put('%s' % name, value)
        function_name = os.path.splitext(os.path.basename(self.module_file))[0]
        func_call = '[{args_out}] = jtlib.modules.{name}({args_in});'.format(
                                    args_out=', '.join(output_names),
                                    name=function_name,
                                    args_in=', '.join(inputs.keys()))
        # Capture standard output and error
        engine.eval("out = evalc('{0}')".format(func_call))
        out = engine.get('out')
        out = re.sub(r'\n$', '', out)  # naicify string
        if out:
            print out  # print to standard output
        for i, name in enumerate(output_names):
            m_out = engine.get('%s' % name)
            logger.debug('dimensions of OUTPUT "{name}": {value}'.format(
                                            name=name, value=m_out.shape))
            logger.debug('type of OUTPUT "{name}": {value}'.format(
                                            name=name, value=type(m_out)))
            logger.debug('dtype of elements of OUTPUT "{name}": {value}'.format(
                                            name=name, value=m_out.dtype))
            output_value = [
                o['value'] for o in self.handles_description['output']
                if o['name'] == name
            ][0]
            # NOTE: Matlab generates numpy array in Fortran order
            self.outputs[output_value] = m_out.copy(order='C')

    def _exec_py_module(self, inputs, output_names):
        logger.debug('importing module: "%s"' % self.module_file)
        module_name = os.path.splitext(os.path.basename(self.module_file))[0]
        module = importlib.import_module('jtlib.modules.%s' % module_name)
        func = getattr(module, module_name)
        logger.debug('evaluating Python function with INPUTS: "%s"',
                     '", "'.join(inputs.keys()))
        py_out = func(**inputs)
        if not output_names:
            return
        if not len(py_out) == len(output_names):
            raise PipelineRunError('number of outputs is incorrect.')
        for i, name in enumerate(output_names):
            # NOTE: The Python function is supposed to return a namedtuple!
            if py_out._fields[i] != name:
                raise PipelineRunError('Incorrect output names.')
            logger.debug('dimensions of OUTPUT "{name}": {value}'.format(
                                            name=name, value=py_out[i].shape))
            logger.debug('type of OUTPUT "{name}": {value}'.format(
                                            name=name, value=type(py_out[i])))
            logger.debug('dtype of elements of OUTPUT "{name}": {value}'.format(
                                            name=name, value=py_out[i].dtype))
            output_value = [
                o['value'] for o in self.handles_description['output']
                if o['name'] == name
            ][0]
            self.outputs[output_value] = py_out[i]

    def _exec_r_module(self, inputs, output_names):
        logger.debug('sourcing module: "%s"' % self.module_file)
        rpy2.robjects.r('source("{0}")'.format(self.module_file))
        rpy2.robjects.numpy2ri.activate()  # enables use of numpy arrays
        function_name = os.path.splitext(os.path.basename(self.module_file))[0]
        func = rpy2.robjects.globalenv['{0}'.format(function_name)]
        logger.debug('evaluating R function with INPUTS: "%s"'
                     % '", "'.join(inputs.keys()))
        # R doesn't have unsigned integer types
        for k, v in inputs.iteritems():
            if isinstance(v, np.ndarray):
                if v.dtype == np.uint16 or v.dtype == np.uint8:
                    logging.debug(
                        'module "%s" input argument "%s": '
                        'convert unsigned integer data type to integer',
                        self.name, k)
                    inputs[k] = v.astype(int)
        args = rpy2.robjects.ListVector({k: v for k, v in inputs.iteritems()})
        base = importr('base')
        r_var = base.do_call(func, args)
        for i, name in enumerate(output_names):
            r_var_np = np.array(r_var.rx2(name))
            logger.debug('dimensions of OUTPUT "{name}": {value}'.format(
                                            name=name, value=r_var_np.shape))
            logger.debug('type of OUTPUT "{name}": {value}'.format(
                                            name=name, value=type(r_var_np)))
            logger.debug('dtype of elements of OUTPUT "{name}": {value}'.format(
                                            name=name, value=r_var_np.dtype))
            output_value = [
                o['value'] for o in self.handles_description['output']
                if o['name'] == name
            ][0]
            self.outputs[output_value] = r_var_np

    def _execute_module(self, inputs, output_names, engine=None):
        if self.language == 'Python':
            self._exec_py_module(inputs, output_names)
        elif self.language == 'Matlab':
            self._exec_m_module(inputs, output_names, engine)
        elif self.language == 'R':
            self._exec_r_module(inputs, output_names)
        else:
            raise PipelineRunError('Language not supported.')

    def prepare_inputs(self, images, upstream_output, data_file, figure_file,
                       job_id, experiment_dir, headless):
        '''
        Prepare input data that will be parsed to the module.

        Parameters
        ----------
        images: Dict[str, numpy.ndarray]
            name of each image and the corresponding pixels array
        upstream_output: dict
            output data generated by modules upstream in the pipeline
        data_file: str
            absolute path to the data file
        figure_file: str
            absolute path to the figure file
        job_id: str
            one-based job identifier number
        experiment_dir: str
            path to experiment directory
        headless: bool
            whether plotting should be disabled

        Note
        ----
        Images are automatically aligned on the fly.

        Warning
        -------
        Be careful when activating plotting because plots are saved as *html*
        files on disk. Their generation requires memory and computation time
        and the files will accumulate on disk.

        Returns
        -------
        dict
            input arguments
        '''
        # Prepare input provided by handles
        inputs = collections.OrderedDict()
        input_names = list()
        for arg in self.handles_description['input']:
            input_names.append(arg['name'])
            if arg['class'] == 'parameter':
                inputs[arg['name']] = arg['value']
            else:
                if arg['value'] in images.keys():
                    # Input pipeline data
                    inputs[arg['name']] = images[arg['value']]
                else:
                    # Upstream pipeline data
                    if arg['value'] is None:
                        continue  # empty value is tolerated for pipeline data
                    pipe_in = {
                        k: v for k, v in upstream_output.iteritems()
                        if k == arg['value']
                    }
                    if arg['value'] not in pipe_in:
                        # TODO: shouldn't this be handled by the checker?
                        raise PipelineRunError(
                                'Incorrect value "%s" for argument "%s" '
                                'in moduel "%s"'
                                % (arg['value'], arg, self.name))
                    inputs[arg['name']] = pipe_in[arg['value']]
        # All additional info potentially required by modules => kwargs
        inputs['data_file'] = data_file
        inputs['figure_file'] = figure_file
        inputs['experiment_dir'] = experiment_dir
        inputs['plot'] = not headless
        inputs['job_id'] = job_id
        return inputs

    def run(self, inputs, engine=None):
        '''
        Execute a module, i.e. evaluate the corresponding function with
        the parsed input arguments as described by `handles`.

        Output has the following format::

            {
                'data': ,               # dict
                'stdout': ,             # str
                'stderr': ,             # str
                'success': ,            # bool
                'error_message': ,      # str
            }

        Parameters
        ----------
        inputs: dict
            input arguments of the module
        engine: matlab_wrapper.matlab_session.MatlabSession, optional
            engine for non-Python languages (default: ``None``)

        Returns
        -------
        dict
            output
        '''
        if self.handles_description['output']:
            output_names = [
                o['name'] for o in self.handles_description['output']
            ]
        else:
            output_names = []

        with CaptureOutput() as output:
            # TODO: the StringIO approach prevents debugging of modules
            try:
                self._execute_module(inputs, output_names, engine)
                success = True
                error = ''
            except Exception as e:
                error = str(e)
                for tb in traceback.format_tb(sys.exc_info()[2]):
                    error += '\n' + tb
                success = False

        stdout = output['stdout']
        sys.stdout.write(stdout)

        stderr = output['stderr']
        stderr += error
        sys.stderr.write(stderr)

        if success:
            output = {
                'data': self.outputs,
                'stdout': stdout,
                'stderr': stderr,
                'success': success,
                'error_message': None
            }
        else:
            output = {
                'data': None,
                'stdout': stdout,
                'stderr': stderr,
                'success': success,
                'error_message': self.build_error_message(
                                        inputs, stdout, stderr)
            }
        return output

    def __str__(self):
        return ':%s: @ <%s>' % (self.name, self.module_file)
