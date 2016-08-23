import os
import sqlite3
import urllib

from utilitites.digits_interface import fetch_digit_models, get_net_info, get_log_info, digits_servers

digits_servers = digits_servers
experiments_folder = os.path.dirname(os.path.realpath(__file__))
db_path = os.path.join(experiments_folder, 'experiments.db')
data_folder = os.path.join(experiments_folder, 'data')

db = sqlite3.Connection(db_path)
net_ids = map(lambda x: x[0], db.execute('''SELECT net_id FROM networks'''))
db_columns_networks = [r[1] for r in db.execute('PRAGMA table_info(networks)')]
db_columns_layers = [r[1] for r in db.execute('PRAGMA table_info(layers)')]

######################
# Fetch DIGITS
######################
for l in fetch_digit_models():
    net_id = l['net_id']
    print net_id

    ######################
    # Update Networks
    ######################
    keys, values = zip(*filter(lambda i: i[0] in db_columns_networks, l.items()))
    if net_id not in net_ids:
        attributes_str = ", ".join(keys)
        values_str = ", ".join(['?' for _ in values])
        net_ids.append(net_id)
        sql_statement = 'insert into networks (%s) values (%s)' % (attributes_str, values_str)
    else:
        attributes_str = ", ".join(map(lambda s: s + "=?", keys))
        sql_statement = "UPDATE networks SET %s WHERE net_id = '%s'" % (attributes_str, net_id)
    cur = db.execute(sql_statement, values)

    ######################
    # Get ProtoTXT files
    ######################
    host = digits_servers[l['node']]
    net_path = os.path.join(data_folder, '%s_net.prototxt' % net_id)
    if not os.path.exists(net_path):
        net_url = host + "/files/" + net_id + "/train_val.prototxt"
        u = urllib.urlopen(net_url)
        if u.getcode() == 404:
            continue
        net_str = u.read();
        open(net_path, 'wb').write(net_str)

    solver_path = os.path.join(data_folder, '%s_solver.prototxt' % net_id)
    if not os.path.exists(solver_path):
        solver_url = host + "/files/" + net_id + "/solver.prototxt"
        u = urllib.urlopen(solver_url)
        if u.getcode() == 404:
            continue
        solver_str = u.read();
        open(solver_path, 'wb').write(solver_str)

    log_path = os.path.join(data_folder, '%s_log.log' % net_id)
    if not os.path.exists(log_path):
        log_url = host + "/files/" + net_id + "/caffe_output.log"
        u = urllib.urlopen(log_url)
        if u.getcode() == 404:
            continue
        log_str = u.read();
        open(log_path, 'wb').write(log_str)

    # print get_net_info(net_id)
    net_info = get_net_info(net_id)
    layers = map(lambda x: x[0], db.execute("SELECT layer_name FROM layers WHERE net_id='%s'" % net_id))
    for l in net_info['layers']:

        l['layer_name'] = l['layer_name'].replace('species', 'clean')

        layer_name = l['layer_name']

        layer_number = l['layer_number']
        keys, values = zip(*filter(lambda i: i[1] is not None and i[0] in db_columns_layers, l.items()))

        values = [str(v) if type(v) == list else v for v in values]

        if layer_name not in layers:
            attributes_str = ", ".join(keys)
            values_str = ", ".join(['?' for _ in values])
            net_ids.append(net_id)
            sql_statement = 'INSERT INTO layers (%s) VALUES (%s)' % (attributes_str, values_str)
        else:
            attributes_str = ", ".join(map(lambda s: s + "=?", keys))
            sql_statement = "UPDATE layers SET %s WHERE net_id = '%s' AND layer_number = %i" % \
                            (attributes_str, net_id, layer_number)

        cur = db.execute(sql_statement, values)
    # Insert

    if net_info.get('batch_size'):
        batch_size = net_info.get('batch_size')
        sql_statement = "UPDATE networks SET batch_size=%i WHERE net_id = '%s'" % (batch_size, net_id)
        cur = db.execute(sql_statement)



        # for l in get_log_info('20160705-214821-0e83')[0].items():
        #     print l

        # print "Batch size: ",get_log_info('20160705-214821-0e83')[1]
        # break

db.commit()

# print get_net_info('20160718-104545-4ac5')
