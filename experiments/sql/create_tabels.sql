CREATE TABLE IF NOT EXISTS networks (
    net_id TEXT,
    PRIMARY KEY (net_id)
    );
ALTER TABLE networks ADD name TEXT;
ALTER TABLE networks ADD batch_size INTEGER;
ALTER TABLE networks ADD solver_type TEXT;
ALTER TABLE networks ADD solver_proto TEXT;
ALTER TABLE networks ADD net_proto TEXT;
ALTER TABLE networks ADD accuracy_max REAL;
ALTER TABLE networks ADD accuracy_val_last REAL;
ALTER TABLE networks ADD learning_rate_train_max REAL;
ALTER TABLE networks ADD progress INTEGER;
ALTER TABLE networks ADD loss_train_last REAL;
ALTER TABLE networks ADD loss_val_last REAL;
ALTER TABLE networks ADD loss_val_min REAL;
ALTER TABLE networks ADD accuracy_val_max REAL;
ALTER TABLE networks ADD learning_rate_train_last REAL;
ALTER TABLE networks ADD learning_rate_train_min REAL;
ALTER TABLE networks ADD loss_train_min REAL;
ALTER TABLE networks ADD caffe_log TEXT;


DROP TABLE IF EXISTS layers;
CREATE TABLE IF NOT EXISTS layers (
    net_id TEXT,
    layer_number INTEGER KEY,
    PRIMARY KEY (net_id, layer_number),
    FOREIGN KEY(net_id) REFERENCES networks(net_id)
    );

ALTER TABLE layers ADD name TEXT;
ALTER TABLE layers ADD params TEXT;
ALTER TABLE layers ADD nparams INTEGER;
ALTER TABLE layers ADD output TEXT;
ALTER TABLE layers ADD trans INTEGER;
ALTER TABLE layers ADD lock INTEGER;
ALTER TABLE layers ADD kernel_size INTEGER;
ALTER TABLE layers ADD stride INTEGER;
ALTER TABLE layers ADD groups INTEGER;
ALTER TABLE layers ADD bottom TEXT;
ALTER TABLE layers ADD top TEXT;
ALTER TABLE layers ADD solver_config TEXT;