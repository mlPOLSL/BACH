class PipelineStrategy(object):
    """
    Blackbox for function to be executed, during execution it searches
    for data point with given IDs, run the function and
    returns own data point with given ID.
    """

    def __init__(self, function=None, inputs_IDs=[],
                 output_ID=None):
        if function:
            self.function = function
        self.inputs_IDs = inputs_IDs
        self.output_ID = output_ID

    def execute(self, input_list):
        inputs = [input.data for input in
                  find_inputs(input_list, self.inputs_IDs)]
        data = self.function(inputs)
        output = PipelineDataPoint(data, self.output_ID)
        return output

    def function(self, inputs):
        return inputs


def find_inputs(list_of_inputs, inputs_IDs):
    inputs = [input for input in list_of_inputs if input.id in inputs_IDs]
    inputs = sorted(inputs, key=lambda input: input.id)
    return inputs


class PipelineDataPoint(object):
    """
    Basic data type in the pipeline, it is used as a container
    to pass data between strategies.
    """

    def __init__(self, data=None, id=None):
        if data is None or id is None:
            raise ValueError(
                "Data point without data, "
                "check if strategies returns correct values")
        self.id = id
        self.data = data


class Pipeline(object):
    """
    Class responsible for queueing the strategies,
    passing data points between strategies
    and executing them in the right order.
    """

    def __init__(self):
        self.strategies = []
        self.data_points = []
        self.data_points_IDs = []

    def add_data_point(self, data_point: PipelineDataPoint):
        self.data_points_IDs.append(data_point.id)
        self.data_points.append(data_point)

    def add_strategy(self, strategy: PipelineStrategy):
        if not all(
                [input in self.data_points_IDs for input in
                 strategy.inputs_IDs]):
            raise ValueError("Given inputs are not present in a pipeline")
        if strategy.output_ID in self.data_points_IDs:
            raise ValueError(
                "Given output ID is already present in a pipeline")
        self.strategies.append(strategy)
        self.data_points_IDs.append(strategy.output_ID)

    def run(self):
        queue = sorted(self.strategies, key=lambda step: step.output_ID)
        for step in queue:
            self.data_points.append(step.execute(self.data_points))
        return self.data_points[-1]
