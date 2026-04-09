import json, re, sys

f = sys.argv[1]
raw = json.dumps(json.load(open(f)), indent=2)

def collapse_lists(s):
    result = []
    i = 0
    while i < len(s):
        if s[i] == '[':
            depth = 1
            j = i + 1
            has_dict = False
            while j < len(s) and depth > 0:
                if s[j] == '[': depth += 1
                elif s[j] == ']': depth -= 1
                elif s[j] == '{': has_dict = True
                j += 1
            if not has_dict:
                c = re.sub(r'\s+', ' ', s[i:j])
                c = c.replace('[ ', '[').replace(' ]', ']')
                # collapse if every element is short
                lines = [l.strip().rstrip(',') for l in s[i:j].split('\n') if l.strip() and l.strip() not in ('[]', '[', ']')]
                if all(len(l) <= 30 for l in lines):
                    result.append(c)
                else:
                    result.append(s[i])
                    i += 1
                    continue
            else:
                result.append(s[i])
                i += 1
                continue
            i = j
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)

open(f.replace('.json', '_formatted.json'), 'w').write(collapse_lists(raw))