
import os
import sqlite3
import matplotlib.pyplot as plt


experiments_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
db_path = os.path.join(experiments_folder, 'experiments.db')
data_folder = os.path.join(experiments_folder, 'data')


db = sqlite3.Connection(db_path)

for r in db.execute(
        '''
        SELECT layers.net_id net_id, networks.name, sum(layers.nparams), accuracy_val_last
        FROM layers INNER JOIN networks ON layers.net_id = networks.net_id
        WHERE lock<1
        GROUP BY layers.net_id
        ORDER BY accuracy_val_last
        '''):
    if r[2] > 4*10**8:
        continue

    if 'raw' in r[1]:
        plt.scatter(r[2], r[3],c='r')
    else:
        plt.scatter(r[2], r[3],c='b')
plt.show()


