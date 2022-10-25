# csv-export2.py

'''
The MIT License (MIT)
Copyright (C) 2022 YasuakiH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def create_csv(default, query, v_csv_filename):
  import csv
  import cx_Oracle
  con = cx_Oracle.connect(default['username'], default['password'], default['service'])
  cursor = con.cursor()
  csv_file = open(v_csv_filename, "w")
  writer = csv.writer(csv_file, delimiter=',', lineterminator="\n", quoting=csv.QUOTE_NONNUMERIC)
  print(query['sql'])
  r = cursor.execute(query['sql'])
  for row in cursor:
    writer.writerow(row)
  cursor.close()
  con.close()
  csv_file.close()

def parse_default_ini(v_config_filename):
  import sys
  import ConfigParser
  config = ConfigParser.RawConfigParser()
  config.read(v_config_filename)
  #
  assert config.get('DB', 'username')
  assert config.get('DB', 'password')
  assert config.get('DB', 'service')
  assert config.get('DB', 'sql')
  assert config.get('DB', 'table_name')
  #
  v_username = config.get('DB', 'username')
  v_password = config.get('DB', 'password')
  v_service = config.get('DB', 'service')
  v_table_name = config.get('DB', 'table_name')
  v_sql = config.get('DB', 'sql')
  #
  return ( {'username':v_username, 'password':v_password, 'service':v_service}, {'table_name':v_table_name, 'sql':v_sql} )

def arg_parse():
  # https://docs.python.org/ja/2.7/library/argparse.html#example
  import argparse
  parser = argparse.ArgumentParser(description='export oracle data as csv.')
  parser.add_argument('--conf', nargs=1, metavar='filename', default=['default.i\ni'], help='a config file (default: default.ini)')
  parser.add_argument('files', metavar='filename', type=str, nargs='+', help='fi\lename(s) for sql')
  args = parser.parse_args()
  print('args.conf = ' + str(args.conf))
  print('args.files = ' + str(args.files))
  return(args)

def main():
  arg_parse()
  default, query = parse_default_ini('default.ini')
  print(str(default))
  print(str(query))
  v_csv_filename = query['table_name'] + '.csv' # "user_tables.csv"
  create_csv(default, query, v_csv_filename)

main()
