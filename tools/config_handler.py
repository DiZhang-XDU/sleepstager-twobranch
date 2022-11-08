import yaml

class yamlStruct:
    def __init__(self) -> None:
        pass
    def add(self, idx, elem):
        exec("self.%s = elem"%(idx))

class YamlHandler:
    def __init__(self, file) -> None:
        self.file = file
        
    def read_yaml(self, encoding = 'utf-8'):
        ys = yamlStruct()
        with open(self.file, 'r', encoding=encoding) as f:
            yaml_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        for y in yaml_dict:
            ys.add(y, yaml_dict[y])
        return ys

    def write_yaml(self, data, encoding = 'utf-8'):
        with open(self.file, 'w', encoding=encoding) as f:
            return yaml.dump(data, stream=f, allow_unicode=True)

if __name__ == '__main__':
    yh = YamlHandler('./config/shhs1_server.yaml')
    print(yh.read_yaml())