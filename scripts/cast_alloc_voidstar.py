#!/usr/bin/env python3
"""
Cast every malloc/calloc/realloc site in the given C-as-C++ sources so the
void* return is explicitly converted to the target pointer type. nvcc's
front-end rejects implicit void*->T* and ignores -fpermissive (lesson L1),
so the C->C++ bridge needs real casts.

Casts NEVER change codegen (a void*->T* pointer cast is a compile-time type
assertion only), so the Serial build stays bit-identical. The compiler is the
final check: any missed or wrong site fails to compile without -fpermissive.

Strategy (handoff / KOKKOS_PORTING_LESSONS L1):
  - declaration   `T *x   = alloc(...)`  -> `T *x   = (T *)alloc(...)`
  - member/id     `lhs    = alloc(...)`  -> `lhs    = (decltype(lhs))alloc(...)`
  - subscript/deref LHS (decltype yields a reference)
                  `a[i]   = alloc(...)`  -> `(std::remove_reference_t<decltype(a[i])>)`
For each alloc call we scan left across whitespace/newlines to the '=' (handles
the one multi-line realloc), then back to the statement boundary (; { }) for LHS.

Usage:  cast_alloc_voidstar.py [--apply] file1.cpp file2.cpp ...
        (no --apply = dry run + report)
"""
import re, sys

alloc_re = re.compile(r'\b(malloc|calloc|realloc)\s*\(')
decl_re  = re.compile(r'^(?P<quals>(?:const\s+)?(?:unsigned\s+|struct\s+)?)'
                      r'(?P<type>[A-Za-z_][A-Za-z0-9_]*)\s*'
                      r'(?P<stars>\*+)\s*'
                      r'[A-Za-z_][A-Za-z0-9_]*$')
BOUND   = set(';{}\n\r')   # LHS stays on the '=' line (the one multi-line realloc has its type+var+'=' on one line)
OPCHARS = set('=!<>+-*/%&|^~')   # char immediately before '=' => compound/compare, skip

def process(path):
    with open(path) as fh:
        text = fh.read()
    edits, report, need_tt, unhandled = [], [], False, []
    for m in alloc_re.finditer(text):
        astart = m.start()
        i = astart - 1
        while i >= 0 and text[i] in ' \t\r\n':
            i -= 1
        if i < 0 or text[i] != '=' or text[i-1] in OPCHARS:
            unhandled.append((path, lineno(text, astart), m.group(1),
                              text[max(0,astart-40):astart].replace('\n','\\n')))
            continue
        eq = i
        k = eq - 1
        while k >= 0 and text[k] not in BOUND:
            k -= 1
        lhs = text[k+1:eq].strip()
        dm = decl_re.match(lhs)
        if dm:
            quals = ' '.join(dm['quals'].split())
            quals = (quals + ' ') if quals else ''
            cast = f"({quals}{dm['type']} {dm['stars']})"
            cat = 'A-decl'
        elif lhs.endswith(']'):
            cast = f"(std::remove_reference_t<decltype({lhs})>)"; cat = 'C-subscript'; need_tt = True
        elif lhs.startswith('*'):
            cast = f"(std::remove_reference_t<decltype({lhs})>)"; cat = 'C-deref'; need_tt = True
        else:
            cast = f"(decltype({lhs}))"; cat = 'B-member'
        edits.append((astart, cast))
        report.append((path, lineno(text, astart), cat, lhs, m.group(1), cast))
    # apply edits right-to-left so positions stay valid
    for pos, cast in sorted(edits, key=lambda e: -e[0]):
        text = text[:pos] + cast + text[pos:]
    if need_tt and '#include <type_traits>' not in text:
        # insert after the first #include line
        mi = re.search(r'^#include[^\n]*\n', text, re.M)
        ins = mi.end() if mi else 0
        text = text[:ins] + '#include <type_traits>\n' + text[ins:]
        report.append((path, 0, 'INCLUDE', '<type_traits>', '', ''))
    return text, report, unhandled

def lineno(text, pos):
    return text.count('\n', 0, pos) + 1

def main():
    apply = '--apply' in sys.argv
    files = [a for a in sys.argv[1:] if a != '--apply']
    all_rep, all_unh, cats = [], [], {}
    for f in files:
        new, rep, unh = process(f)
        all_rep += rep; all_unh += unh
        for r in rep:
            cats[r[2]] = cats.get(r[2], 0) + 1
        if apply:
            with open(f, 'w') as fh:
                fh.write(new)
    print(f"=== sites cast: {sum(1 for r in all_rep if r[2] != 'INCLUDE')} "
          f"(across {len(files)} files){'  [APPLIED]' if apply else '  [DRY RUN]'} ===")
    for c in sorted(cats):
        print(f"  {c:14s} {cats[c]}")
    print("\n=== Cat C (reference-trap) + INCLUDE sites (full): ===")
    for r in all_rep:
        if r[2].startswith('C-') or r[2] == 'INCLUDE':
            print(f"  {r[0]}:{r[1]:<5} {r[2]:12s} LHS={r[3]!r}  cast={r[5]}")
    print("\n=== distinct Cat A casts: ===")
    for cast in sorted({r[5] for r in all_rep if r[2] == 'A-decl'}):
        print(f"  {cast}")
    if all_unh:
        print(f"\n=== !!! UNHANDLED ({len(all_unh)}) — alloc not preceded by a plain '=': ===")
        for u in all_unh:
            print(f"  {u[0]}:{u[1]} {u[2]}  ...ctx: {u[3]!r}")
    else:
        print("\n=== UNHANDLED: none ===")

if __name__ == '__main__':
    main()
