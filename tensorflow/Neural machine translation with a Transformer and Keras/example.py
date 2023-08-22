import numpy as np
import tensorflow as tf

pt = tf.constant([[0, 0]])
en = tf.constant([[0]])

from PositionalEmbedding import PositionalEmbedding

embed_pt = PositionalEmbedding(vocab_size=2, d_model=512)
embed_en = PositionalEmbedding(vocab_size=2, d_model=512)

pt_emb = embed_pt(pt)
en_emb = embed_en(en)

print(pt_emb.shape)
print(en_emb.shape)
# print(pt_emb)
# print(en_emb)
print(en_emb._keras_mask)


from Transformer import Transformer

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=2,
    target_vocab_size=2,
    dropout_rate=dropout_rate
)



from CustomSchedule import CustomSchedule

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)



transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

transformer.fit([[(pt, en)]],
                [[en]],
                epochs=20)


