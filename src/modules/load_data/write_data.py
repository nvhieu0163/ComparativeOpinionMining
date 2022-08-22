
def export_to_file(export_file_path, tokens, ner_tags, stc_idxs):
    print('x', stc_idxs)
    with open(export_file_path, "w", encoding='utf-8') as f:
        for i in range(0, len(tokens)):
            tags = ner_tags[i]
            toks = tokens[i]
            # idxs = stc_idxs[i]
            print('t',tags)
            if len(tokens) > 0:
                f.write(
                    # str(idxs)
                    # + "\t"
                    str(len(toks))
                    + "\t"
                    + "\t".join(toks)
                    + "\t"
                    + "\t".join(map(str, tags))
                    + "\n"
                )