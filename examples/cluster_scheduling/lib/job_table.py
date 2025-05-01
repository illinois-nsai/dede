from .job_template import JobTemplate


def job(i):
    model = f'job_{i}'
    working_directory = ''
    command = ''
    num_steps_arg = ''
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg)


JobTable = []
for i in range(2587):
    JobTable.append(job(i))
