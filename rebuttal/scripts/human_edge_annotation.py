#!/usr/bin/env python3
"""Minimal blinded web UI for independent reasoning-edge annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import secrets
import sys
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.dag.graph import analyze_topology_projection  # noqa: E402
from src.capacity_profiles import json_capacity_profiles  # noqa: E402


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TopoPRM edge annotation</title>
<style>
:root { color-scheme: light; --ink:#17202a; --muted:#5d6875; --line:#d8dde3; --soft:#f4f6f8; --accent:#146c43; --warn:#a33a2b; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); background:#fff; font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { border-bottom:1px solid var(--line); padding:14px 22px; display:flex; align-items:center; justify-content:space-between; gap:12px; }
h1 { margin:0; font-size:18px; font-weight:600; letter-spacing:0; }
main { width:min(1080px,calc(100% - 28px)); margin:20px auto 40px; }
.meta,.actions { display:flex; flex-wrap:wrap; align-items:center; gap:10px; }
.meta { color:var(--muted); margin-bottom:12px; }
.progress { height:6px; background:#e7eaee; margin-bottom:20px; }
.progress > span { display:block; height:100%; background:var(--accent); }
section { padding:18px 0; border-bottom:1px solid var(--line); }
h2 { margin:0 0 10px; font-size:16px; font-weight:600; letter-spacing:0; }
.problem { white-space:pre-wrap; }
.step { display:grid; grid-template-columns:38px minmax(0,1fr); gap:10px; padding:11px 0; border-top:1px solid #edf0f2; }
.step-id { font-variant-numeric:tabular-nums; color:var(--accent); font-weight:600; }
.step-text { white-space:pre-wrap; overflow-wrap:anywhere; }
.edge-target { margin:14px 0; }
.target-label { font-weight:600; margin-bottom:7px; }
.checks { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:7px 14px; }
label { display:flex; align-items:flex-start; gap:7px; }
input[type=checkbox],input[type=radio] { margin-top:4px; accent-color:var(--accent); }
textarea { width:100%; min-height:74px; padding:9px; border:1px solid var(--line); border-radius:4px; font:inherit; resize:vertical; }
button { min-height:38px; padding:8px 13px; border:1px solid #9fa8b2; border-radius:4px; background:#fff; color:var(--ink); font:inherit; cursor:pointer; }
button.primary { color:#fff; background:var(--accent); border-color:var(--accent); }
button:disabled { opacity:.45; cursor:not-allowed; }
.status { min-height:24px; color:var(--muted); }
.error { color:var(--warn); }
.rubric { color:var(--muted); max-width:900px; }
@media (max-width:640px) { header { align-items:flex-start; } main { width:min(100% - 20px,1080px); } .checks { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header><h1>Reasoning dependency annotation</h1><div id="slot"></div></header>
<main>
  <div class="meta"><span id="counter"></span><span id="saved"></span></div>
  <div class="progress"><span id="bar"></span></div>
  <section><h2>Rubric</h2><div class="rubric">Mark i -> j only when the fact or result established in step i is directly needed to derive or justify step j. A later step may support an earlier unsupported assertion; mark that backward edge instead of forcing the relation forward. Do not mark mere word overlap or a purely transitive dependency. Flag unusable segmentation instead of guessing.</div></section>
  <section><h2>Problem</h2><div id="problem" class="problem"></div></section>
  <section><h2>Reasoning steps</h2><div id="steps"></div></section>
  <section><h2>Direct support edges</h2><div id="edges"></div></section>
  <section>
    <h2>Quality check</h2>
    <label><input id="bad-seg" type="checkbox"> The step segmentation is unusable</label>
    <div class="edge-target"><div class="target-label">Confidence</div><div class="checks">
      <label><input type="radio" name="confidence" value="low"> Low</label>
      <label><input type="radio" name="confidence" value="medium" checked> Medium</label>
      <label><input type="radio" name="confidence" value="high"> High</label>
    </div></div>
    <label for="comment">Optional note</label><textarea id="comment"></textarea>
  </section>
  <section class="actions"><button id="prev">Previous</button><button id="save" class="primary">Save and next</button><button id="skip">Next without saving</button><span id="status" class="status"></span></section>
</main>
<script>
const pathBits=location.pathname.split('/').filter(Boolean);
const slot=pathBits[0]||'';
const access=pathBits[1]||'';
let index=Number(new URLSearchParams(location.search).get('index')||0);
let state=null;
const $=id=>document.getElementById(id);
function esc(s){const d=document.createElement('div');d.textContent=s??'';return d.innerHTML;}
async function load(){
  $('status').textContent='Loading...';
  const res=await fetch(`/api/state?slot=${encodeURIComponent(slot)}&access=${encodeURIComponent(access)}&index=${index}`);
  if(!res.ok){$('status').className='status error';$('status').textContent=await res.text();return;}
  state=await res.json(); index=state.index;
  $('slot').textContent=`Annotator ${state.slot.toUpperCase()}`;
  $('counter').textContent=`Item ${state.index+1} of ${state.total}`;
  $('saved').textContent=`Saved ${state.saved_count}/${state.total}`;
  $('bar').style.width=`${100*state.saved_count/state.total}%`;
  $('problem').textContent=state.record.question;
  $('steps').innerHTML=state.record.steps.map((s,i)=>`<div class="step"><div class="step-id">[${i}]</div><div class="step-text">${esc(s)}</div></div>`).join('');
  const selected=new Set((state.annotation.edges||[]).map(e=>`${e[0]}-${e[1]}`));
  let groups=[];
  for(let j=0;j<state.record.steps.length;j++){
    let boxes=[];
    for(let i=0;i<state.record.steps.length;i++){
      if(i===j) continue;
      const direction=i<j?'forward':'backward';
      boxes.push(`<label><input class="edge" type="checkbox" data-i="${i}" data-j="${j}" ${selected.has(`${i}-${j}`)?'checked':''}> [${i}] -> [${j}] (${direction})</label>`);
    }
    groups.push(`<div class="edge-target"><div class="target-label">Sources directly supporting step [${j}]</div><div class="checks">${boxes.join('')}</div></div>`);
  }
  $('edges').innerHTML=groups.join('');
  $('bad-seg').checked=Boolean(state.annotation.segmentation_unusable);
  document.querySelectorAll('input[name=confidence]').forEach(x=>x.checked=x.value===(state.annotation.confidence||'medium'));
  $('comment').value=state.annotation.comment||'';
  $('prev').disabled=state.index===0; $('skip').disabled=state.index===state.total-1;
  $('save').textContent=state.index===state.total-1?'Save':'Save and next';
  $('status').className='status'; $('status').textContent=state.annotation.saved?'Saved annotation loaded':'';
  history.replaceState(null,'',`/${slot}/${access}?index=${state.index}`);
}
async function save(){
  const edges=[...document.querySelectorAll('.edge:checked')].map(x=>[Number(x.dataset.i),Number(x.dataset.j)]);
  const payload={slot,access,record_id:state.record.record_id,edges,segmentation_unusable:$('bad-seg').checked,confidence:document.querySelector('input[name=confidence]:checked').value,comment:$('comment').value.trim()};
  $('status').textContent='Saving...';
  const res=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  if(!res.ok){$('status').className='status error';$('status').textContent=await res.text();return;}
  if(index<state.total-1) index++; await load();
}
$('prev').onclick=()=>{if(index>0){index--;load();}};
$('skip').onclick=()=>{if(state&&index<state.total-1){index++;load();}};
$('save').onclick=save;
load().catch(e=>{$('status').className='status error';$('status').textContent=String(e);});
</script>
</body></html>
"""


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TopoPRM trace audit</title>
<style>
:root{color-scheme:light;--ink:#17202a;--muted:#66717d;--line:#d7dde3;--soft:#f4f6f8;--green:#176b4d;--red:#b24436;--blue:#2d5f91;--amber:#9a6700}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#fff;font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}header{height:58px;padding:0 22px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:16px}h1{margin:0;font-size:18px;font-weight:650;letter-spacing:0}button,a.button{min-height:34px;padding:7px 11px;border:1px solid #aeb6bf;border-radius:4px;color:var(--ink);background:#fff;font:inherit;text-decoration:none;cursor:pointer}button.primary{color:#fff;border-color:var(--green);background:var(--green)}main{width:min(1380px,calc(100% - 28px));margin:18px auto 36px}.toolbar,.nav,.legend{display:flex;flex-wrap:wrap;align-items:center;gap:8px}.toolbar{justify-content:space-between;margin-bottom:14px}.muted{color:var(--muted)}.summary{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border:1px solid var(--line);margin-bottom:14px}.summary-item{padding:9px 12px;border-right:1px solid var(--line)}.summary-item:last-child{border-right:0}.summary-item span{display:block;color:var(--muted);font-size:11px}.summary-item strong{font:650 17px/1.3 ui-monospace,SFMono-Regular,Consolas,monospace}.grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(340px,.75fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.main-pane{padding:18px 22px 18px 0;border-right:1px solid var(--line);min-width:0}.side-pane{padding:18px 0 18px 22px;min-width:0}h2{margin:0 0 10px;font-size:15px;font-weight:650;letter-spacing:0}.problem{margin-bottom:9px;white-space:pre-wrap}.answer{margin-bottom:16px;padding:8px 10px;border-left:3px solid var(--blue);background:var(--soft)}.graph-wrap{width:100%;max-width:100%;min-height:390px;overflow:auto;contain:inline-size;border:1px solid var(--line);background:#fbfcfd}svg{display:block;min-width:760px;width:100%;height:390px}.edge-forward{stroke:var(--green)}.edge-backward{stroke:var(--red);stroke-dasharray:7 5}.edge-cycle{stroke:var(--amber);stroke-width:3;stroke-dasharray:3 4}.node rect{fill:#fff;stroke:#8c98a4;stroke-width:1.2}.node text{fill:var(--ink);font-size:12px}.node .id{fill:var(--blue);font-weight:700}.legend{margin-top:9px;color:var(--muted)}.swatch{width:24px;height:0;border-top:3px solid var(--green)}.swatch.backward{border-color:var(--red);border-top-style:dashed}.swatch.cycle{border-color:var(--amber);border-top-style:dotted}.edge-ledger,.variant-table{width:100%;margin-top:15px;border-collapse:collapse;font-size:12px}.edge-ledger th,.edge-ledger td,.variant-table th,.variant-table td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.edge-ledger th,.variant-table th{color:var(--muted);font-weight:600}.edge-ledger td:first-child{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.edge-ledger .risk{color:var(--red);font-weight:650}.edge-ledger .cycle-risk{color:var(--amber);font-weight:650}.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));border-top:1px solid var(--line);border-left:1px solid var(--line)}.metric{min-height:74px;padding:10px 12px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.metric .label{color:var(--muted);font-size:12px}.metric .value{margin-top:4px;font:650 23px/1.1 ui-monospace,SFMono-Regular,Consolas,monospace}.formula{margin-top:16px;padding:12px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.formula code{display:block;margin:6px 0;white-space:pre-wrap;color:#24364a;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}.reward-row{display:grid;grid-template-columns:96px 1fr 46px;gap:9px;align-items:center;margin:8px 0}.bar{height:8px;background:#e8ebee}.bar span{display:block;height:100%;background:var(--blue)}.steps{margin-top:16px;border-top:1px solid var(--line)}.step{display:grid;grid-template-columns:34px minmax(0,1fr);gap:9px;padding:9px 0;border-bottom:1px solid #edf0f2}.step-id{color:var(--blue);font-weight:700;font-variant-numeric:tabular-nums}.step-text{overflow-wrap:anywhere}.annotation-links{margin-top:17px;padding-top:14px;border-top:1px solid var(--line)}.status{min-height:20px;color:var(--muted)}@media(max-width:900px){.summary{grid-template-columns:repeat(2,minmax(0,1fr))}.summary-item{border-bottom:1px solid var(--line)}.grid{grid-template-columns:1fr}.main-pane{padding-right:0;border-right:0;border-bottom:1px solid var(--line)}.side-pane{padding-left:0}}
.edge-ledger{table-layout:fixed}.edge-ledger th,.edge-ledger td{overflow-wrap:anywhere}
.capacity-preview{margin-top:16px;padding:14px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.capacity-preview p{margin:0 0 10px}.segment{display:flex;gap:0;margin-bottom:12px}.segment button{flex:1;border-radius:0}.segment button:first-child{border-radius:4px 0 0 4px}.segment button:last-child{border-radius:0 4px 4px 0}.segment button+button{margin-left:-1px}.source-controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 12px;margin:10px 0}.source-control{display:flex;align-items:center;gap:7px;color:var(--muted)}.source-control input[type=range]{min-width:0;width:100%;accent-color:var(--blue)}.source-control code{width:42px;text-align:right}.reward-total{display:flex;align-items:baseline;justify-content:space-between;margin:10px 0 4px}.reward-total strong:last-child{font:700 25px/1 ui-monospace,SFMono-Regular,Consolas,monospace}.reward-detail{white-space:pre-wrap;color:#24364a;font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
</style></head><body>
<header><h1>TopoPRM trace audit</h1><div class="muted">online topology verification · <span id="provenance"></span></div></header>
<main><div id="summary" class="summary"></div><div class="toolbar"><div><strong id="counter"></strong> <span id="record" class="muted"></span></div><div class="nav"><span id="variants" class="nav"></span><button id="prev">Previous</button><button id="sample" class="primary">Sample trace</button><button id="next">Next</button></div></div>
<div class="grid"><section class="main-pane"><h2>Reasoning trace</h2><div id="problem" class="problem"></div><div id="answer" class="answer"></div><h2>Recovered support graph</h2><div class="graph-wrap"><svg id="graph" role="img" aria-label="Reasoning dependency graph"></svg></div><div class="legend"><span class="swatch"></span> forward support <span class="swatch backward"></span> backward support <span class="swatch cycle"></span> cycle-participating <span id="edge-summary"></span></div><table class="edge-ledger"><thead><tr><th>Edge</th><th>Confidence</th><th>Direction</th><th>Cycle</th><th>Evidence type</th></tr></thead><tbody id="edge-ledger"></tbody></table><div id="steps" class="steps"></div></section>
<aside class="side-pane"><h2>Auditable raw-graph terms</h2><div id="metrics" class="metrics"></div><div class="formula"><strong>Coverage-adjusted calculation</strong><code id="formula"></code><div id="reward-bars"></div></div><div class="capacity-preview"><h2>Choquet reward preview</h2><p class="muted">Trace-measured direction and acyclicity are combined with explicit audit assumptions below. Candidate capacity only; no profile has been approved for Stage II.</p><div id="capacity-segment" class="segment"></div><div class="source-controls"><label class="source-control"><input id="assume-outcome" type="checkbox" checked> outcome <code id="outcome-value">1.000</code></label><label class="source-control"><input id="assume-format" type="checkbox" checked> format <code id="format-value">1.000</code></label><label class="source-control">continuity <input id="assume-continuity" type="range" min="0" max="1" step="0.01" value="1"><code id="continuity-value">1.000</code></label></div><div class="reward-total"><strong>R_total</strong><strong id="reward-total"></strong></div><div id="reward-detail" class="reward-detail"></div></div><h2 style="margin-top:16px">Same trace under order controls</h2><table class="variant-table"><thead><tr><th>Order</th><th>q_dir</th><th>q_acyc</th><th>Back.</th><th>Cycle</th></tr></thead><tbody id="variant-metrics"></tbody></table><div class="annotation-links"><h2>Independent human check</h2><p class="muted">Three blinded copies use independently shuffled trace order. Labels are mapped back to source-step coordinates for agreement and edge F1.</p><div id="annotators" class="nav"></div></div><p id="status" class="status"></p></aside></div></main>
<script>
const bits=location.pathname.split('/').filter(Boolean),access=bits[1]||'',params=new URLSearchParams(location.search);let index=Number(params.get('index')||0),variant=params.get('variant')||'original',state=null,capacityProfile='balanced',auditAssumptions={outcome:1,format:1,continuity:1};const $=id=>document.getElementById(id),fmt=x=>Number(x).toFixed(3);function esc(s){const d=document.createElement('div');d.textContent=s??'';return d.innerHTML}function marker(id,color){return `<marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="${color}"/></marker>`}
function renderRewardPreview(metrics){const profiles=state.capacity_profiles,profile=profiles[capacityProfile],values={...auditAssumptions,direction:metrics.direction_score,acyclicity:metrics.acyclicity_score};$('capacity-segment').innerHTML=Object.keys(profiles).map(name=>`<button class="${name===capacityProfile?'primary':''}" data-profile="${name}">${name==='balanced'?'Balanced':'Structure-forward'}</button>`).join('');$('capacity-segment').querySelectorAll('button').forEach(button=>button.onclick=()=>{capacityProfile=button.dataset.profile;renderRewardPreview(metrics)});$('assume-outcome').checked=Boolean(auditAssumptions.outcome);$('assume-format').checked=Boolean(auditAssumptions.format);$('assume-continuity').value=String(auditAssumptions.continuity);$('outcome-value').textContent=fmt(auditAssumptions.outcome);$('format-value').textContent=fmt(auditAssumptions.format);$('continuity-value').textContent=fmt(auditAssumptions.continuity);let total=0,terms=[];for(const [name,weight] of Object.entries(profile.singletons)){const contribution=weight*values[name];total+=contribution;terms.push(`${weight.toFixed(3)} ${name} = ${contribution.toFixed(3)}`)}for(const item of profile.interactions){const [left,right]=item.sources,contribution=item.weight*Math.min(values[left],values[right]);total+=contribution;terms.push(`${item.weight.toFixed(3)} min(${left}, ${right}) = ${contribution.toFixed(3)}`)}$('reward-total').textContent=fmt(total);$('reward-detail').textContent=`z = (${fmt(values.outcome)}, ${fmt(values.format)}, ${fmt(values.direction)}, ${fmt(values.acyclicity)}, ${fmt(values.continuity)})\n${terms.join('\n')}`;$('assume-outcome').onchange=event=>{auditAssumptions.outcome=event.target.checked?1:0;renderRewardPreview(metrics)};$('assume-format').onchange=event=>{auditAssumptions.format=event.target.checked?1:0;renderRewardPreview(metrics)};$('assume-continuity').oninput=event=>{auditAssumptions.continuity=Number(event.target.value);renderRewardPreview(metrics)}}
function drawGraph(record){const svg=$('graph'),steps=record.steps,edges=record.edges,W=Math.max(760,svg.clientWidth||760),H=390,cols=Math.min(4,Math.max(2,Math.ceil(Math.sqrt(steps.length)))),rows=Math.ceil(steps.length/cols),nodeW=Math.min(190,(W-70)/cols-24),nodeH=62,points=[];for(let i=0;i<steps.length;i++){const col=i%cols,row=Math.floor(i/cols);points.push({x:35+col*((W-70)/cols),y:35+row*((H-100)/Math.max(rows-1,1))})}let out=`<defs>${marker('arrow-f','#176b4d')}${marker('arrow-b','#b24436')}${marker('arrow-c','#9a6700')}</defs>`;for(const e of edges){const a=points[e.source],b=points[e.target];if(!a||!b)continue;const backward=e.source>e.target,x1=a.x+nodeW/2,y1=a.y+nodeH/2,x2=b.x+nodeW/2,y2=b.y+nodeH/2,bend=backward?-48:28,edgeClass=e.cycle_participating?'edge-cycle':(backward?'edge-backward':'edge-forward'),arrow=e.cycle_participating?'arrow-c':(backward?'arrow-b':'arrow-f');out+=`<path d="M${x1} ${y1}Q${(x1+x2)/2} ${(y1+y2)/2+bend} ${x2} ${y2}" fill="none" stroke-width="2" class="${edgeClass}" marker-end="url(#${arrow})"><title>${esc(e.dep_type)}: ${e.source}->${e.target}; confidence ${fmt(e.weight)}${e.cycle_participating?'; cycle-participating':''}</title></path>`}for(let i=0;i<steps.length;i++){const p=points[i],label=steps[i].length>28?steps[i].slice(0,25)+'...':steps[i];out+=`<g class="node"><rect x="${p.x}" y="${p.y}" width="${nodeW}" height="${nodeH}" rx="3"/><text x="${p.x+10}" y="${p.y+20}" class="id">S${i+1}</text><text x="${p.x+10}" y="${p.y+42}">${esc(label)}</text><title>${esc(steps[i])}</title></g>`}svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=out}
function render(){const r=state.record,m=state.metrics,s=state.dataset_summary,p=state.prediction_provenance;variant=state.variant;$('provenance').textContent=`${p.label} · ${p.sha256.slice(0,12)}`;$('summary').innerHTML=[['Frozen traces',s.records],['Original mean q_dir',fmt(s.original.direction_score)],['Original mean q_acyc',fmt(s.original.acyclicity_score)],['Reverse delta q_dir',`${s.reverse_delta_direction>=0?'+':''}${fmt(s.reverse_delta_direction)}`],['Reverse delta q_acyc',`${s.reverse_delta_acyclicity>=0?'+':''}${fmt(s.reverse_delta_acyclicity)}`]].map(([k,v])=>`<div class="summary-item"><span>${k}</span><strong>${v}</strong></div>`).join('');$('counter').textContent=`Trace ${state.index+1} / ${state.total}`;$('record').textContent=`${r.record_id} · ${r.source} · ${r.band} · ${variant}`;$('variants').innerHTML=state.available_variants.map(v=>`<button class="${v===variant?'primary':''}" data-variant="${v}">${v[0].toUpperCase()+v.slice(1)}</button>`).join('');$('variants').querySelectorAll('button').forEach(b=>b.onclick=()=>{variant=b.dataset.variant;load()});$('problem').textContent=r.question;$('answer').innerHTML=`<strong>Reference final answer</strong> ${esc(r.final_answer||'not available')}`;drawGraph(r);$('edge-summary').textContent=`${r.edges.length} retained · ${m.backward_edges} backward · ${m.cycle_edges} cyclic`;$('edge-ledger').innerHTML=r.edges.length?r.edges.map(e=>`<tr><td>S${e.source+1} &rarr; S${e.target+1}</td><td>${fmt(e.weight)}</td><td class="${e.orientation==='backward'?'risk':''}">${e.orientation}</td><td class="${e.cycle_participating?'cycle-risk':''}">${e.cycle_participating?'yes':'no'}</td><td>${esc(e.dep_type)}</td></tr>`).join(''):'<tr><td colspan="5" class="muted">No retained support edges</td></tr>';$('steps').innerHTML=r.steps.map((step,i)=>`<div class="step"><span class="step-id">S${i+1}</span><span class="step-text">${esc(step)}</span></div>`).join('');const specs=[['Direction q_dir',m.direction_score],['Acyclicity q_acyc',m.acyclicity_score],['Coverage c_H',m.dependency_coverage],['Forward mass d_H',m.raw_direction],['Cycle mass u_H',m.cycle_edge_mass],['Pairs kept / abstained',`${m.evaluated_pairs}/${m.abstained_pairs}`]];$('metrics').innerHTML=specs.map(([k,v])=>`<div class="metric"><div class="label">${k}</div><div class="value">${typeof v==='number'?fmt(v):v}</div></div>`).join('');$('formula').textContent=`q_dir = ${fmt(m.dependency_coverage)} x ${fmt(m.raw_direction)} + (1 - ${fmt(m.dependency_coverage)}) / 2 = ${fmt(m.direction_score)}\nq_acyc = ${fmt(m.dependency_coverage)} x (1 - ${fmt(m.cycle_edge_mass)}) + (1 - ${fmt(m.dependency_coverage)}) / 2 = ${fmt(m.acyclicity_score)}\nindependent audit channels; Stage-II capacity pending confirmation`;const bars=[['direction',m.direction_score],['acyclicity',m.acyclicity_score],['coverage',m.dependency_coverage]];$('reward-bars').innerHTML=bars.map(([k,v])=>`<div class="reward-row"><span>${k}</span><span class="bar"><span style="width:${100*v}%"></span></span><code>${fmt(v)}</code></div>`).join('');renderRewardPreview(m);$('variant-metrics').innerHTML=state.variant_metrics.map(x=>`<tr><td>${x.variant}</td><td>${fmt(x.direction_score)}</td><td>${fmt(x.acyclicity_score)}</td><td>${x.backward_edges}</td><td>${x.cycle_edges}</td></tr>`).join('');$('annotators').innerHTML=['a','b','c'].map(slot=>`<a class="button" href="/${slot}/${access}?index=${state.annotation_indices[slot]}">Annotator ${slot.toUpperCase()} · ${state.annotation_progress[slot]}/${state.total}</a>`).join('');$('prev').disabled=state.index===0;$('next').disabled=state.index===state.total-1;$('status').textContent=`Frozen prediction snapshot: ${p.label} · sha256 ${p.sha256.slice(0,12)}. Direction and acyclicity remain independent; Stage-II capacity is pending confirmation.`;history.replaceState(null,'',`/dashboard/${access}?index=${state.index}&variant=${variant}`)}
async function load(){$('status').textContent='Loading trace...';const res=await fetch(`/api/dashboard?access=${encodeURIComponent(access)}&index=${index}&variant=${encodeURIComponent(variant)}`);if(!res.ok){$('status').textContent=await res.text();return}state=await res.json();index=state.index;render()}$('prev').onclick=()=>{if(index>0){index--;load()}};$('next').onclick=()=>{if(state&&index<state.total-1){index++;load()}};$('sample').onclick=()=>{if(state){index=Math.floor(Math.random()*state.total);load()}};load().catch(e=>$('status').textContent=String(e));
</script></body></html>
"""


def _edge_payload(edge: object) -> dict[str, object] | None:
    if isinstance(edge, dict):
        try:
            source, target = int(edge["source"]), int(edge["target"])
        except (KeyError, TypeError, ValueError):
            return None
        return {
            "source": source,
            "target": target,
            "weight": max(0.0, float(edge.get("weight", edge.get("confidence", 1.0)))),
            "dep_type": str(edge.get("dep_type") or edge.get("edge_type") or "support"),
        }
    if isinstance(edge, list) and len(edge) == 2:
        return {"source": int(edge[0]), "target": int(edge[1]), "weight": 1.0, "dep_type": "support"}
    return None


def _topology_dashboard(row: dict) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    n_steps = len(row.get("steps", []))
    edges = [payload for edge in row.get("extractor_edges", []) if (payload := _edge_payload(edge))]
    edges = [edge for edge in edges if 0 <= edge["source"] < n_steps and 0 <= edge["target"] < n_steps]
    metrics, cyclic_ids = analyze_topology_projection(
        range(n_steps),
        [
            (int(edge["source"]), int(edge["target"]), float(edge["weight"]))
            for edge in edges
        ],
    )
    cyclic = [
        edge
        for edge in edges
        if (int(edge["source"]), int(edge["target"])) in cyclic_ids
    ]
    for edge in edges:
        source, target = int(edge["source"]), int(edge["target"])
        edge["orientation"] = "forward" if source < target else "backward"
        edge["cycle_participating"] = (source, target) in cyclic_ids
    return edges, {
        "raw_direction": metrics["raw_direction"],
        "direction_score": metrics["direction_score"],
        "dependency_coverage": metrics["dependency_coverage"],
        "cycle_edge_mass": metrics["cycle_edge_mass"],
        "acyclicity_score": metrics["acyclicity_score"],
        "backward_edges": sum(edge["source"] > edge["target"] for edge in edges),
        "cycle_edges": len(cyclic),
        "candidate_pairs": int(row.get("candidate_unordered_pairs", n_steps * max(n_steps - 1, 0) // 2)),
        "evaluated_pairs": int(row.get("evaluated_pairs", len(edges))),
        "abstained_pairs": int(row.get("abstained_pairs", 0)),
    }


class AnnotationStore:
    def __init__(
        self,
        pack: Path,
        output_dir: Path,
        slots: list[str],
        access: str,
        predictions: Path | None = None,
    ) -> None:
        source_records = [json.loads(line) for line in pack.open() if line.strip()]
        dashboard_records = [dict(row) for row in source_records]
        self.dashboard_variants: dict[str, dict[str, dict]] = {}
        if predictions is not None:
            source_by_id = {str(row["record_id"]): row for row in dashboard_records}
            for line in predictions.open():
                if not line.strip():
                    continue
                pred = json.loads(line)
                source_id = str(pred.get("source_record_id") or pred["record_id"])
                source = source_by_id.get(source_id)
                if source is None:
                    continue
                variant = str(pred.get("order_variant") or "original")
                order = pred.get("step_order") or list(range(len(source.get("steps", []))))
                if sorted(order) != list(range(len(source.get("steps", [])))):
                    raise ValueError(f"invalid step order for {source_id}:{variant}")
                inverse = {source_index: display_index for display_index, source_index in enumerate(order)}
                display_edges = pred.get("display_edges")
                if display_edges is None:
                    display_edges = []
                    for edge in pred.get("edges", []):
                        payload = dict(edge) if isinstance(edge, dict) else {"source": edge[0], "target": edge[1]}
                        payload["source"] = inverse[int(payload["source"])]
                        payload["target"] = inverse[int(payload["target"])]
                        display_edges.append(payload)
                row = dict(source)
                row["steps"] = [source["steps"][source_index] for source_index in order]
                row["extractor_edges"] = display_edges
                row["candidate_unordered_pairs"] = pred.get("candidate_unordered_pairs", 0)
                row["evaluated_pairs"] = pred.get("evaluated_pairs", 0)
                row["abstained_pairs"] = pred.get("abstained_pairs", 0)
                row["prediction_order_variant"] = variant
                self.dashboard_variants.setdefault(source_id, {})[variant] = row
            missing = [str(row["record_id"]) for row in dashboard_records if str(row["record_id"]) not in self.dashboard_variants]
            if missing:
                raise ValueError(f"predictions missing {len(missing)} pack records; first={missing[0]}")
        else:
            self.dashboard_variants = {
                str(row["record_id"]): {"original": row} for row in dashboard_records
            }
        aggregate: dict[str, dict[str, float]] = {}
        for variant in ("original", "reversed", "interleaved"):
            values = [
                _topology_dashboard(variants[variant])[1]
                for variants in self.dashboard_variants.values()
                if variant in variants
            ]
            if values:
                aggregate[variant] = {
                    "direction_score": sum(float(item["direction_score"]) for item in values) / len(values),
                    "acyclicity_score": sum(float(item["acyclicity_score"]) for item in values) / len(values),
                    "coverage": sum(float(item["dependency_coverage"]) for item in values) / len(values),
                }
        original = aggregate.get("original", {"direction_score": 0.0, "acyclicity_score": 0.0, "coverage": 0.0})
        reversed_metrics = aggregate.get("reversed", original)
        self.dashboard_summary = {
            "records": len(dashboard_records),
            "original": original,
            "reverse_delta_direction": reversed_metrics["direction_score"] - original["direction_score"],
            "reverse_delta_acyclicity": reversed_metrics["acyclicity_score"] - original["acyclicity_score"],
        }
        self.dashboard_records = dashboard_records
        if predictions is None:
            self.prediction_provenance = {
                "label": "pack extractor edges",
                "filename": None,
                "sha256": hashlib.sha256(pack.read_bytes()).hexdigest(),
            }
        else:
            name = predictions.name
            label = (
                "corrected-segmentation"
                if "correctedseg" in name
                else "pre-correction"
            )
            self.prediction_provenance = {
                "label": label,
                "filename": name,
                "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            }
        self.records = [self._with_order_control(row, index) for index, row in enumerate(source_records)]
        self.by_id = {row["record_id"]: row for row in self.records}
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.slots = set(slots)
        self.access = access
        self.orders: dict[str, list[str]] = {}
        self.annotation_indices: dict[str, dict[str, int]] = {}
        for slot in slots:
            ids = list(self.by_id)
            seed = int.from_bytes(hashlib.sha256(f"topoprm-human-v1:{slot}".encode()).digest()[:8], "big")
            random.Random(seed).shuffle(ids)
            self.orders[slot] = ids
            self.annotation_indices[slot] = {
                self.by_id[record_id]["source_record_id"]: index
                for index, record_id in enumerate(ids)
            }

    @staticmethod
    def _with_order_control(row: dict, index: int) -> dict:
        """Balance original, reversed, and interleaved step orders.

        Human labels should test semantic direction rather than reward the
        encoder for copying presentation order.  Each source record appears
        once, under one deterministic order condition, so annotation burden is
        unchanged and conditions remain matched across annotator slots.
        """
        out = dict(row)
        steps = list(row.get("steps", []))
        order = list(range(len(steps)))
        variant = ("original", "reversed", "interleaved")[index % 3]
        if variant == "reversed":
            order.reverse()
        elif variant == "interleaved":
            order = order[::2] + order[1::2]
        source_id = str(row["record_id"])
        opaque_suffix = hashlib.sha256(f"{source_id}:{variant}".encode()).hexdigest()[:8]
        out["source_record_id"] = source_id
        out["record_id"] = f"{source_id}_{opaque_suffix}"
        out["order_variant"] = variant
        out["step_order"] = order
        out["steps"] = [steps[i] for i in order]
        return out

    def authorized(self, slot: str, access: str) -> bool:
        return slot in self.slots and secrets.compare_digest(access, self.access)

    def dashboard_authorized(self, access: str) -> bool:
        return secrets.compare_digest(access, self.access)

    def path(self, slot: str) -> Path:
        return self.output_dir / f"annotator_{slot}.json"

    def load(self, slot: str) -> dict[str, dict]:
        path = self.path(slot)
        if not path.exists():
            return {}
        with path.open() as handle:
            payload = json.load(handle)
        return payload.get("annotations", {})

    def save(self, slot: str, record_id: str, annotation: dict) -> None:
        annotations = self.load(slot)
        annotations[record_id] = annotation
        payload = {"schema_version": 1, "annotator_slot": slot, "annotations": annotations}
        fd, tmp_name = tempfile.mkstemp(prefix=f".{slot}-", suffix=".json", dir=self.output_dir)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(tmp_name, self.path(slot))
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)


def handler_factory(store: AnnotationStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TopoPRMAnnotation/1.0"

        def log_message(self, fmt: str, *args) -> None:
            print(f"[{self.log_date_time_string()}] {self.client_address[0]} {fmt % args}", flush=True)

        def send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_bytes(json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def send_error_text(self, status: HTTPStatus, message: str) -> None:
            self.send_bytes(message.encode(), "text/plain; charset=utf-8", status)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self.send_json({"status": "ok", "records": len(store.records)})
                return
            if parsed.path == "/api/dashboard":
                query = parse_qs(parsed.query)
                access = query.get("access", [""])[0]
                if not store.dashboard_authorized(access):
                    self.send_error_text(HTTPStatus.FORBIDDEN, "Invalid dashboard link")
                    return
                try:
                    index = max(0, min(int(query.get("index", ["0"])[0]), len(store.dashboard_records) - 1))
                except ValueError:
                    index = 0
                source = store.dashboard_records[index]
                source_id = str(source["record_id"])
                variants = store.dashboard_variants[source_id]
                variant = query.get("variant", ["original"])[0]
                if variant not in variants:
                    variant = next(iter(variants))
                row = variants[variant]
                edges, metrics = _topology_dashboard(row)
                variant_metrics = []
                for variant_name in ("original", "reversed", "interleaved"):
                    if variant_name not in variants:
                        continue
                    _, variant_values = _topology_dashboard(variants[variant_name])
                    variant_metrics.append({"variant": variant_name, **variant_values})
                self.send_json({
                    "index": index,
                    "total": len(store.dashboard_records),
                    "variant": variant,
                    "available_variants": [
                        name for name in ("original", "reversed", "interleaved") if name in variants
                    ],
                    "record": {
                        "record_id": source_id,
                        "source": row.get("source", ""),
                        "band": row.get("band", ""),
                        "question": row.get("question", ""),
                        "final_answer": row.get("final_answer", ""),
                        "steps": row.get("steps", []),
                        "edges": edges,
                    },
                    "metrics": metrics,
                    "variant_metrics": variant_metrics,
                    "capacity_profiles": json_capacity_profiles(),
                    "prediction_provenance": store.prediction_provenance,
                    "dataset_summary": store.dashboard_summary,
                    "annotation_indices": {
                        slot: store.annotation_indices[slot][source_id]
                        for slot in sorted(store.slots)
                    },
                    "annotation_progress": {
                        slot: len(store.load(slot)) for slot in sorted(store.slots)
                    },
                })
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                slot = query.get("slot", [""])[0]
                access = query.get("access", [""])[0]
                if not store.authorized(slot, access):
                    self.send_error_text(HTTPStatus.FORBIDDEN, "Invalid annotation link")
                    return
                try:
                    index = max(0, min(int(query.get("index", ["0"])[0]), len(store.records) - 1))
                except ValueError:
                    index = 0
                annotations = store.load(slot)
                record_id = store.orders[slot][index]
                row = store.by_id[record_id]
                annotation = annotations.get(record_id, {})
                self.send_json({
                    "slot": slot,
                    "index": index,
                    "total": len(store.records),
                    "saved_count": len(annotations),
                    "record": {"record_id": record_id, "question": row["question"], "steps": row["steps"]},
                    "annotation": {**annotation, "saved": record_id in annotations},
                })
                return
            bits = parsed.path.strip("/").split("/")
            if len(bits) == 2 and bits[0] == "dashboard" and store.dashboard_authorized(bits[1]):
                self.send_bytes(DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
                return
            if len(bits) == 2 and store.authorized(bits[0], bits[1]):
                self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
                return
            self.send_error_text(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/save":
                self.send_error_text(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("Invalid payload size")
                payload = json.loads(self.rfile.read(length))
                slot = str(payload.get("slot", ""))
                access = str(payload.get("access", ""))
                if not store.authorized(slot, access):
                    self.send_error_text(HTTPStatus.FORBIDDEN, "Invalid annotation link")
                    return
                record_id = str(payload.get("record_id", ""))
                row = store.by_id.get(record_id)
                if row is None:
                    raise ValueError("Unknown record")
                edges = []
                seen = set()
                for edge in payload.get("edges", []):
                    if not isinstance(edge, list) or len(edge) != 2:
                        raise ValueError("Malformed edge")
                    source, target = int(edge[0]), int(edge[1])
                    if not 0 <= source < len(row["steps"]) or not 0 <= target < len(row["steps"]) or source == target:
                        raise ValueError("Edge outside valid step range")
                    if (source, target) not in seen:
                        seen.add((source, target))
                        edges.append([source, target])
                confidence = str(payload.get("confidence", "medium"))
                if confidence not in {"low", "medium", "high"}:
                    raise ValueError("Invalid confidence")
                annotation = {
                    "record_id": record_id,
                    "source_record_id": row["source_record_id"],
                    "order_variant": row["order_variant"],
                    "step_order": row["step_order"],
                    "edges": sorted(edges),
                    "segmentation_unusable": bool(payload.get("segmentation_unusable", False)),
                    "confidence": confidence,
                    "comment": str(payload.get("comment", ""))[:1000],
                }
                store.save(slot, record_id, annotation)
                self.send_json({"saved": True})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_error_text(HTTPStatus.BAD_REQUEST, str(exc))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--slots", default="a,b,c")
    parser.add_argument("--access-token", default=os.getenv("TOPOPRM_ANNOTATION_TOKEN", ""))
    args = parser.parse_args()
    slots = [slot.strip().lower() for slot in args.slots.split(",") if slot.strip()]
    if not slots or not args.access_token:
        parser.error("at least one slot and --access-token are required")
    store = AnnotationStore(args.pack, args.output_dir, slots, args.access_token, args.predictions)
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(store))
    print(f"Serving {len(store.records)} records on {args.host}:{args.port}; slots={','.join(slots)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
