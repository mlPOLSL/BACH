import unittest
from Utilities.Pipeline.pipeline import PipelineDataPoint, PipelineStrategy, \
    Pipeline


class TestPipelineDataPoint(unittest.TestCase):
    def test_if_raises_without_data_and_ID(self):
        self.assertRaises(ValueError, PipelineDataPoint)

    def test_if_raises_without_ID(self):
        self.assertRaises(ValueError, PipelineDataPoint, 1)

    def test_if_raises_without_data(self):
        self.assertRaises(ValueError, PipelineDataPoint, None, 1)

    def test_if_object_is_created_with_data_and_id(self):
        data_point = PipelineDataPoint(1, 1)
        self.assertIsInstance(data_point, PipelineDataPoint)

    def test_if_object_i_created_with_proper_data(self):
        data_point = PipelineDataPoint(1, 2)
        self.assertEqual(data_point.data, 1)

    def test_if_object_i_created_with_proper_id(self):
        data_point = PipelineDataPoint(1, 2)
        self.assertEqual(data_point.id, 2)


class TestPipelineStrategy(unittest.TestCase):
    def setUp(self):
        self.data_points_list = [PipelineDataPoint("data_point" + str(x), x)
                                 for x
                                 in range(0, 4)]

    def test_that_default_function_passes_inputs(self):
        strategy = PipelineStrategy(inputs_IDs=[1], output_ID=3)
        self.assertEqual("data_point1",
                         strategy.execute(self.data_points_list).data[0])

    def test_that_custom_function_replaces_default_one(self):
        def custom_function(input):
            return "test_passed"

        strategy = PipelineStrategy(custom_function, inputs_IDs=[1],
                                    output_ID=3)
        self.assertEqual(strategy.execute(self.data_points_list).data,
                         "test_passed")


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data_points_list = [PipelineDataPoint("data_point" + str(x), x)
                                 for x
                                 in range(0, 4)]
        self.pipeline = Pipeline()

    def test_that_raises_when_incorrect_input_id_is_passed(self):
        strategy1 = PipelineStrategy(inputs_IDs=[0], output_ID=4)
        self.assertRaises(ValueError, self.pipeline.add_strategy,
                          strategy1)

    def test_that_raises_when_output_id_is_already_in_data_points(self):
        self.pipeline.add_data_point(self.data_points_list[0])
        strategy1 = PipelineStrategy(inputs_IDs=[0], output_ID=0)
        self.assertRaises(ValueError, self.pipeline.add_strategy,
                          strategy1)

    def test_that_data_points_are_added_correctly(self):
        self.pipeline.add_data_point(self.data_points_list[0])
        self.assertIn(self.data_points_list[0], self.pipeline.data_points)

    def test_that_data_points_are_passed_correctly(self):
        strategy1 = PipelineStrategy(inputs_IDs=[0], output_ID=1)
        strategy2 = PipelineStrategy(inputs_IDs=[1], output_ID=2)
        self.pipeline.add_data_point(self.data_points_list[0])
        self.pipeline.add_strategy(strategy1)
        self.pipeline.add_strategy(strategy2)
        self.assertEqual(self.pipeline.run().data, [["data_point0"]])
