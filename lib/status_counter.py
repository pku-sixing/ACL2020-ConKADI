from lib import utils
from collections import defaultdict
import time
class Status:

    def __init__(self, report_per_steps=100):

        self.start_time = -1
        self.last_report_time = -1
        self.current_time = -1
        self.last_report_step = 0
        self.current_step = -1
        self.report_per_steps = report_per_steps

        self.value_dict = defaultdict(float)

    def add_record(self, report_dict, step, epoch):

        if self.start_time == -1: # First report
            self.start_time = time.time()
            self.last_report_time = time.time()
            self.current_time = time.time()
            self.last_report_step = step - 1
            self.current_step = step
        else:
            self.current_time = time.time()
            self.current_step = step

        # update
        for key in report_dict:
            self.value_dict[key] += report_dict[key]

        if self.current_step - self.last_report_step  >= self.report_per_steps:
            num_steps = self.current_step - self.last_report_step
            num_time = self.current_time - self.last_report_time
            step_time = num_time / num_steps

            summary = []
            for key in report_dict:
                if key == 'lr' or key == 'learning_rate':
                    summary.append('%s=%f' % (key, self.value_dict[key] / num_steps))
                else:
                    summary.append('%s=%.2f' % (key, self.value_dict[key] / num_steps))
            utils.print_out('#[E%d/Step%d]  Training Summary: interval steps: %d, step_per_time=%.2f' % (epoch, self.current_step, num_steps, step_time))
            utils.print_out('\t'.join(summary))

            self.last_report_step = step
            self.last_report_time = time.time()
            self.value_dict = defaultdict(float)

