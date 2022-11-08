from test_twoLoss import tester
from tools.config_handler import YamlHandler

if __name__ == '__main__':
    yh = YamlHandler('./config/ran_all_server.yaml')
    cfg = yh.read_yaml()

    tester(cfg, alpha = .5)