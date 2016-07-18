import re


def parse(iterator):
    d = dict()
    for l in iterator:
        key = None
        m = re.search('(\w+) {', l)
        if m:
            key = m.groups()[0]
            value = parse(iterator)
        elif '}' in l:
            return d
        for m in re.finditer('(\w+): (".+?"|[0-9]*\.[0-9]+|\w+)', l):
            key = m.groups(0)[0]
            value = m.groups(0)[1]
        if not key:
            continue
        if key in d:
            d[key].append(value)
        else:
            d[key] = [value]
    return d


def net2str(d, depth=0):
    indent = depth * "  "
    lines = []
    for k, vs in d.items():
        for v in vs:
            if type(v) == dict:
                lines.append(indent + "%s {\n" % k)
                lines.append(net2str(v, depth + 1))
                lines.append(indent + "}\n")
            else:
                lines.append(indent + "%s: %s\n" % (k, v))
    return "".join(lines)

#
# def rewrite_data_layer(net):
#     for l in net['layer']:
#         if l['name'][0] == '"data"':
#             l['type'] = ['"ImageData"']
#             l['ntop'] = [2]
#             l['new_height'] = [256]
#             l['new_width'] = [256]
#             if 'data_param' in l:
#                 del l['data_param']
#             print l
#     return net
