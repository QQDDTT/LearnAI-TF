import tensorflow as tf
import yaml
from models.model import build_model
from data.dataloader import load_data

# 读取配置
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
epochs = config["epochs"]

# 加载数据
(x_train, y_train), (x_test, y_test) = load_data()

# 构建模型
model = build_model(config["model"])

# 编译
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 训练
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# 保存模型
model.save("models/saved_model")
