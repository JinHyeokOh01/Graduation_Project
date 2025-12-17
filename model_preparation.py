import tensorflow as tf
import numpy as np
from pathlib import Path

print("="*70)
print("Quantized CPU Block (Keras 3 호환)")
print("="*70)

base = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3),
    alpha=1.0,
    include_top=True,
    weights='imagenet'
)

# Block 8 찾기
block8_idx = None
for idx, layer in enumerate(base.layers):
    if 'conv_pw_8' in layer.name:
        block8_idx = idx
        print(f"✓ Block 8: [{idx}] {layer.name}")
        break

# Block 9-end
block9_input = tf.keras.Input(shape=(14, 14, 512))
x = block9_input
for idx in range(block8_idx + 1, len(base.layers)):
    x = base.layers[idx](x)

block_9_end = tf.keras.Model(inputs=block9_input, outputs=x)
print(f"✓ Block 9-end 완료!")

# export() 사용!
output_dir = Path('hef_output')
saved_model_dir = output_dir / 'cpu_block_9_end_saved'
block_9_end.export(saved_model_dir)  # ← export() 사용!
print(f"✓ SavedModel: {saved_model_dir}")

# Representative dataset
calib_data = []
for i in range(100):
    data = np.random.rand(1, 14, 14, 512).astype(np.float32)
    data = (data - 0.5) * 2
    calib_data.append(data)

def representative_dataset():
    for data in calib_data:
        yield [data]

# Quantization
print(f"\nQuantization...")
converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_quant = converter.convert()

tflite_path = output_dir / 'cpu_blocks_9_end_uint8.tflite'
tflite_path.write_bytes(tflite_quant)

print(f"\n✓ 완료!")
print(f"  {tflite_path} ({len(tflite_quant)/1024/1024:.2f} MB)")

# 테스트
print(f"\n{'='*70}")
print("테스트")
print(f"{'='*70}")

block_1_8 = tf.keras.Model(inputs=base.input, outputs=base.layers[block8_idx].output)

test_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
test_img = (test_img - 0.5) * 2

full_pred = base.predict(test_img, verbose=0)
block8_out = block_1_8.predict(test_img, verbose=0)

interp = tf.lite.Interpreter(model_path=str(tflite_path))
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()

block8_uint8 = ((block8_out + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
interp.set_tensor(inp[0]['index'], block8_uint8)
interp.invoke()
quant_out = interp.get_tensor(out[0]['index'])

full_top1 = np.argmax(full_pred)
quant_top1 = np.argmax(quant_out.astype(np.float32))

print(f"Full: {full_top1}, Quantized: {quant_top1}")
print(f"{'✓✓✓ MATCH!' if full_top1 == quant_top1 else '⚠️  Quantization loss'}")

print(f"\n{'='*70}")
print(f"scp -P 13373 {tflite_path} icns@163.180.117.25:~/transfer_to_icns/")
print(f"{'='*70}")
