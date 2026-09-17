import subprocess,dynamic_yaml

def load_config(config_path):
# Load the configuration file using PyYAML
    with open(config_path, 'r') as f:
        cfg = dynamic_yaml.load(f)
    return cfg

def main():
    cfg = load_config('simulate_ventilation/ventilation.yaml')
    ie_ratio = cfg.i_to_e_ratio

    for subject,weight in zip(list(cfg.subject_list),list(cfg.weight_list)):
        subprocess.run(["python","simulate_ventilation/define_init_volume.py",str(subject),str(weight),str(ie_ratio)])

if __name__ == '__main__':
    main()