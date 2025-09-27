import yaml
from data.dataloader import load_data
from models.model import build_model
import tensorflow as tf
import os

def main():
    # 1️⃣ 读取配置文件
    config_path = os.path.join("config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    batch_size = config.get("batch_size", 64)
    learning_rate = config.get("learning_rate", 0.001)
    epochs = config.get("epochs", 10)
    model_cfg = config.get("model", {"hidden_units": 128, "dropout": 0.2})

    # 2️⃣ 加载数据
    (x_train, y_train), (x_test, y_test) = load_data()

    # 3️⃣ 构建模型
    model = build_model(model_cfg)

    # 4️⃣ 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 5️⃣ 训练模型
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test)
    )

    # 6️⃣ 保存模型
    saved_model_dir = os.path.join("models", "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)
    model.save(saved_model_dir)
    print(f"模型已保存到 {saved_model_dir}")

if __name__ == "__main__":
    main()
