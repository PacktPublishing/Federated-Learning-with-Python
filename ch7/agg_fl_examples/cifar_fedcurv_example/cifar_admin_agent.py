from stadle import AdminAgent
from stadle import BaseModelConvFormat
from stadle.lib.entity.model import BaseModel
from stadle.lib.util import client_arg_parser

from vgg import VGG as Model


def get_base_model():
    return BaseModel("PyTorch-CIFAR10-Model", Model('VGG16'), BaseModelConvFormat.pytorch_format)


if __name__ == '__main__':
    args = client_arg_parser()
    config_path = r'config/config_admin_agent.json'

    admin_agent = AdminAgent(config_file=config_path, simulation_flag=args.simulation,
                             aggregator_ip_address=args.aggregator_ip, reg_port=args.reg_port,
                             exch_port=args.exch_port, model_path=args.model_path, base_model=get_base_model(),
                             agent_running=args.agent_running)

    admin_agent.preload()
    admin_agent.initialize()
