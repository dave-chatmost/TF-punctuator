import sys
import os
import shutil


def compute_interpolate(line1, line2, weight):
    post = {}
    l1 = line1.split()
    l2 = line2.split()
    assert l1[0] == '<unk>' or l2[0] == '<unk>' or l1[0] == l2[0], "mismatch word"
    j = 1
    for i in range(5): # 5 classes
        post[l1[j]] = float(l1[j+1])
        j += 2
    j = 1
    for i in range(5):
        post[l2[j]] = weight * post[l2[j]] + (1-weight) * float(l2[j+1])
        j += 2
    return post


def get_puncts(post1, post2, weight):
    puncts = []
    with open(post1, 'r') as p1, open(post2, 'r') as p2:
        for line1, line2 in zip(p1, p2):
            post = compute_interpolate(line1, line2, weight)
            punct = max(zip(post.values(), post.keys()))[1]
            if punct == "*noevent*":
                punct = ' '
            puncts.append(punct)
        #print(puncts)
    return puncts


def write_punctuations(unpunc_file, puncts, out_file):
    print("Punctuating", unpunc_file)
    i = 0
    with open(unpunc_file, 'r') as inpf, open(out_file, 'w') as outf:
        for line in inpf:
            for word in line.split():
                if puncts[i] == " ":
                    outf.write("%s " % word)
                else:
                    outf.write("%s %s " % (word, puncts[i]))
                i += 1
            outf.write("\n")
    print("Put result in", out_file)


def interpolate(unpunc_file, post_file1, post_file2, out_file, weight):
    print("Interpolating %s and %s" % (post_file1, post_file2))
    puncts = get_puncts(post_file1, post_file2, weight)
    write_punctuations(unpunc_file, puncts, out_file)


if __name__ == "__main__":
    UNPUNC_FILE = sys.argv[1]
    POST_A_FILE = sys.argv[2]
    POST_B_FILE = sys.argv[3]
    OUT_FILE = sys.argv[4]
    WEIGHT = float(sys.argv[5])
    interpolate(UNPUNC_FILE, POST_A_FILE, POST_B_FILE, OUT_FILE, WEIGHT)