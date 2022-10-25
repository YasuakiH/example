# csv-export1.py

'''
The MIT License (MIT)
Copyright (C) 2022 YasuakiH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

'''
-------------default.ini----------------
[DB]
foodir: %(dir)s/whatever
dir=frob
long: this value continues
   in the next line

username=SYSTEM
password=manager
service=orcl
sql=
 select
  username,
  user_id,
  to_char(created,'YYYY/MM/DD') as created
 from all_users
 order by created
'''
# ----------------- csv-export.py --------------
def create_csv(v_username, v_password, v_service, v_sql, v_csv_filename):
  import csv
  import cx_Oracle
  con = cx_Oracle.connect(v_username, v_password, v_service)
  cursor = con.cursor()
  # csv_file = open("user_tables.csv", "w")
  csv_file = open(v_csv_filename, "w")
  writer = csv.writer(csv_file, delimiter=',', lineterminator="\n", quoting=csv.QUOTE_NONNUMERIC)
  # r = cursor.execute("select username, user_id, to_char(created,'YYYY/MM/DD') created from all_users order by created")
  r = cursor.execute(v_sql)
  for row in cursor:
    writer.writerow(row)
  cursor.close()
  con.close()
  csv_file.close()

def parse_default_ini(v_config_filenam):
  import sys
  import ConfigParser
  config = ConfigParser.RawConfigParser()
  config.read(v_config_filenam)
  #
  assert config.get('DB', 'username')
  assert config.get('DB', 'password')
  assert config.get('DB', 'service')
  assert config.get('DB', 'sql')
  #
  v_username = config.get('DB', 'username')
  v_password = config.get('DB', 'password')
  v_service = config.get('DB', 'service')
  v_sql = config.get('DB', 'sql')
  #
  return (v_username, v_password, v_service, v_sql)

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
  default = parse_default_ini('default.ini')
  v_csv_filename = "user_tables.csv"
  create_csv(default[0], default[1], default[2], default[3], v_csv_filename)

main()
