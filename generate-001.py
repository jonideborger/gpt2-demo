import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

gpt2.generate(
    sess,
    length=300,
    seed=None,
    temperature=0.7,
    prefix="Help me, god."
)