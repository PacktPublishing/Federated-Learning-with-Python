# Aggregation algorithm + FL examples

The aggregation algorithms FedProx and FedCurv are implemented within an FL example using the CIFAR-10 classification task.


Both the original and modified FL examples are implemented using STADLE.
To run these examples, first install `stadle-client` using the command
```
pip install stadle-client
```
Next, go to [stadle.ai](stadle.ai) and create a new project (sign up for an account if you haven't yet).  Follow the usage guide to start an aggregator
and modify the aggregator ip and port in the FL example config file based on the listed connection information.

Once this is done, you can run the FL example.  First, upload the VGG-16 model using the following command:
```
stadle upload-model --config_path config.agent.json
```

After the model upload completes, you can start agent(s) using the following command:
```
python fl_training.py --agent_name {unique name for agent}
```

Make sure to use unique names for each agent when running multiple agents from the same machine.


For the FedCurv example, we include arguments to control how the local datasets used by each agent are skewed.
The CIFAR-10 classes are `'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'`

To specify the classes to be skewed towards in the constructed local dataset, include the class names as flags for the command.
In addition, the arguments `sel_count` and `def_count` can be used to set the percentage of training examples to keep for the selected (specified)
classes and default (other) classes, respectively.

For example, the command
```
python fl_training.py --sel_count 1.0 --def_count 0.1 --airplane --bird --truck
```
will keep all of the training examples with class airplane, bird, or truck, and will delete 90% of the training examples for the other seven classes.