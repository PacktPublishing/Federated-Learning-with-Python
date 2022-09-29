import argparse
import json

from ibmfl.party.party import Party

parser = argparse.ArgumentParser()
parser.add_argument("party_id", type=int)

args = parser.parse_args()
party_id = args.party_id

with open('party_config.json') as cfg_file:
    party_config = json.load(cfg_file)

party_config['connection']['info']['port'] += party_id
party_config['connection']['info']['id'] += f'_{party_id}'
party_config['data']['info']['client_id'] = party_id

party = Party(config_dict=party_config)
party.start()
party.register_party()

