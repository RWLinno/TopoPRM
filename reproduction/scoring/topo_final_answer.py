"""Score the final-answer format explicitly requested by the Topo eval prompt.

The frozen matcher recognizes boxes and final-answer prose, but omitted closed
<answer> tags. This adapter adds that contract without using incidental numbers
from reasoning. Original predictions and online reward computations stay intact.
"""
from pathlib import Path
import importlib.util,re

SOURCE=Path(__file__).resolve().parents[1]/'code/src/eval/math_scoring.py'
spec=importlib.util.spec_from_file_location('topo_frozen_math_scoring',SOURCE)
matcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(matcher)
VERSION='final-answer-contract-v3-20261108'

def score(response,gold):
 text=str(response)
 matches=list(re.finditer(r'<answer>(.*?)</answer>',text,re.S))
 if matches and not text[matches[-1].end():].strip():
  candidate=matches[-1].group(1).strip()
  if not candidate:return False,'empty_answer_tag'
  # Currency-prefixed numeric answers are distinct from LaTeX math delimiters.
  if re.fullmatch(r'\$\s*[-+]?\d[\d,]*(?:\.\d+)?',candidate):candidate=candidate[1:].strip()
  return bool(matcher.verify_answer_equivalence(candidate,str(gold))),'terminal_answer_tag'
 # An echoed template tag or a tag followed by units does not invalidate an
 # otherwise valid boxed answer under the original frozen scoring contract.
 return bool(matcher.verify_math_response(text,str(gold))),'frozen_box_or_explicit_final'
