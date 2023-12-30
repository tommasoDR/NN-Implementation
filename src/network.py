from utilities import data_checks
from functions import loss_functions
from functions import metric_functions
from layer import Layer


class Network:
    def __init__(
        self,
        input_dimension,
        num_layers,
        layers_sizes,
        layers_activation_funcs,
        loss_func,
        metric_func,
        weight_init_type,
        weight_init_range=None,
    ):
        """
        Initializes the network
        :param input_dimension: The dimension of the input
        :param num_layers: The number of layers in the network (excluding the input layer)
        :param layers_sizes: The number of units in each layer (excluding the input layer)
        :param layers_activation_funcs: The activation functions for each hidden layer
        :param loss_func: The loss function of the network
        :param metric_func: The metric function of the network
        :param weight_init_type: The type of weight initialization
        :param weight_init_range: The range of the weight initialization, if required by the weight initialization type
        """
        self.input_dim = input_dimension
        self.output_dim = layers_sizes[-1]
        self.num_layers = num_layers
        self.layer_sizes = layers_sizes
        self.layers_activation_funcs = layers_activation_funcs
        self.loss = loss_functions.loss_funcs[loss_func]
        self.metric = metric_functions.metric_funcs[metric_func]
        self.weight_init_type = weight_init_type
        self.weight_init_range = weight_init_range

        # Check parameters
        try:
            self.parameters = {
                "num_layers": num_layers,
                "layers_sizes": layers_sizes,
                "layers_activation_funcs": layers_activation_funcs,
                "loss_func": loss_func,
                "metric_func": metric_func,
                "weight_init_type": weight_init_type,
                "weight_init_range": weight_init_range,
            }
            data_checks.check_param(self.parameters)
        except Exception as e:
            print(e)
            exit(1)

        # Initialize layers
        self.layers = []
        layer_input_dim = input_dimension

        for i in range(self.num_layers):
            self.layers.append(
                Layer(
                    input_dim=layer_input_dim,
                    num_unit=layers_sizes[i],
                    activation_func=layers_activation_funcs[i],
                    weight_init_type=weight_init_type,
                    weight_init_range=weight_init_range,
                )
            )
            layer_input_dim = layers_sizes[i]


    def foward_pass(self, inputs):
        """
        Performs a forward pass through the network
        :param input: The input of the network
        :return: The output of the output layer of the network
        """
        outputs = inputs
        for layer in self.layers:
            outputs = layer.foward_pass(outputs)
        return outputs


    def backpropagation(self, dErr_dOut):
        """
        Performs backpropagation
        :param dErr_dOut: The derivative of the error with respect to the output of the network
        :return: The gradients of the network
        """
        for layer in reversed(self.layers):
            dErr_dOut = layer.backward_pass(dErr_dOut)


    def compute_loss(self, inputs, targets, loss_function):
        """
        Calculates the loss of the network
        :param inputs: The inputs of the network
        :param targets: The expected outputs of the network
        :param loss_function: The loss function to use for the calculation
        :return: The loss of the network
        """
        try:
            data_checks.check_sets(inputs, self.input_dim, targets, self.output_dim)
        except Exception as e:
            print(e)
            exit(1)

        loss = loss_functions.loss_funcs[loss_function]
        outputs = self.forward_pass(inputs)

        return loss.function(outputs, targets)


    def compute_loss_derivative(self, outputs, targets):
        """
        Calculates the derivative of the loss of the network
        :param inputs: The inputs of the network
        :param targets: The expected outputs of the network
        :return: The derivative of the loss of the network
        """
        return self.loss.derivative(outputs, targets)


    def predict(self, inputs):
        """
        Calculates the outputs of the network
        :param inputs: The inputs of the network
        :return: The outputs of the network
        """
        try:
            data_checks.check_sets(inputs, self.input_dim)
        except Exception as e:
            print(e)
            exit(1)

        return self.foward_pass(inputs)


    def evaluate(self, inputs, targets):
        """
        Evaluates the loss and the metric of the network on the given inputs and targets
        :param inputs: The inputs of the network
        :param targets: The expected outputs of the network 
        :return: The loss and the metric of the network
        """
        try:
            outputs = self.foward_pass(inputs)
        except Exception as e:
            print(e)
            exit(1)

        loss = self.loss.function(outputs, targets)
        metric = self.metric.function(
            outputs, targets, self.layers_activation_funcs[-1]
        )

        return loss, metric


    def set_loss(self, loss_func):
        """
        Sets the loss function of the network
        :param loss_func: The loss function to use
        """
        loss = {}
        loss["loss_func"] = loss_func
        try:
            data_checks.check_param(loss)
        except Exception as e:
            print(e)
            exit(1)
        self.loss = loss_functions.loss_funcs[loss_func]


    def set_metric(self, metric_func):
        """
        Sets the metric function of the network
        :param metric_func: The metric function to use
        """
        metric = {}
        metric["metric_func"] = metric_func
        try:
            data_checks.check_param(metric)
        except Exception as e:
            print(e)
            exit(1)
        self.metric = metric_functions.metric_funcs[metric_func]