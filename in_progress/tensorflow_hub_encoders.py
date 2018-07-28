import tensorflow as tf
import tensorflow_hub as hub


TEXT_ENCODER_MODULES={'google/nnlm-en-dim50/1':\
									"https://tfhub.dev/google/nnlm-en-dim50/1",
					  'google/nnlm-en-dim128/1':\
									"https://tfhub.dev/google/nnlm-en-dim128/1",
					  'google/nnlm-en-dim50-with-normalization/1':\
									"https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1",
					  'google/nnlm-en-dim128-with-normalization/1':\
									"https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1",
					  'google/random-nnlm-en-dim128/1':\
					                "https://tfhub.dev/google/random-nnlm-en-dim128/1",
					  'google/universal-sentence-encoder/1':\
					                "https://tfhub.dev/google/universal-sentence-encoder/1",
					  'google/universal-sentence-encoder-large/1':\
					                "https://tfhub.dev/google/universal-sentence-encoder-large/1"
					  'google/elmo/1':\
					                "https://tfhub.dev/google/elmo/1"}


def get_hub_embedding(name):
	with tf.Graph().as_default():
	  embed = hub.Module(name)
	  embeddings = embed
	  
	#   with tf.Session() as sess:
	#     sess.run(tf.global_variables_initializer())
	#     sess.run(tf.tables_initializer())
	#     print(sess.run(embeddings))
