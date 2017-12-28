class PipelineStrategy(object):
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
        output = PipelineOutput(data, self.output_ID)
        return output

    def function(self, inputs):
        print("Strategy execution inputs: {}, output: {}".format(
            self.inputs_IDs, self.output_ID))
        return PipelineOutput(self.inputs_IDs, self.output_ID)


def find_inputs(list_of_inputs, inputs_IDs):
    inputs = [input for input in list_of_inputs if input.id in inputs_IDs]
    return inputs


class PipelineOutput(object):
    def __init__(self, data=None, id=None):
        if data == None and id == None:
            raise ValueError(
                "Data point without data, check if strategies returns correct values")
        self.id = id
        self.data = data


class Pipeline(object):
    def __init__(self):
        self.strategies = []
        self.data_points = []
        self.data_points_IDs = []

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


a = PipelineStrategy(output_ID=1)
b = PipelineStrategy(inputs_IDs=[1], output_ID=2)
c = PipelineStrategy(inputs_IDs=[2], output_ID=3)
d = PipelineStrategy(inputs_IDs=[3], output_ID=4)
e = PipelineStrategy(inputs_IDs=[3], output_ID=5)
f = PipelineStrategy(inputs_IDs=[4, 5], output_ID=6)

pipeline = Pipeline()
pipeline.add_strategy(a)
pipeline.add_strategy(b)
pipeline.add_strategy(c)
pipeline.add_strategy(d)
pipeline.add_strategy(e)
pipeline.add_strategy(f)
pipeline.run()
