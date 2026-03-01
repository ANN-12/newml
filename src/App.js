import React, { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";

const TARGET   = "the quick brown fox jumps over the lazy dog";
const API_BASE = "http://localhost:5000";

// ── Helpers (all timing in ms) ──────────────────────────
const avg = (a) => a.length ? a.reduce((x,y)=>x+y,0)/a.length : 0;
const std = (a) => {
  if (a.length<2) return 0;
  const m=avg(a);
  return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/a.length);
};
const cv = (a) => { const m=avg(a); return m>0?std(a)/m:0; };

function entropy(lst) {
  if (lst.length<2) return 0;
  const mn=Math.min(...lst), mx=Math.max(...lst);
  if (mx===mn) return 0;
  const bins=10, w=(mx-mn)/bins;
  const hist=new Array(bins).fill(0);
  lst.forEach(v=>{ const b=Math.min(bins-1,Math.floor((v-mn)/w)); hist[b]++; });
  return -hist.reduce((s,c)=>{ if(!c) return s; const p=c/lst.length; return s+p*Math.log2(p); },0);
}

// ── Digraph/Trigraph maps (only pairs in sentence) ──────
const DIGRAPH_MAP = {
  "th":"dg_th","he":"dg_he","qu":"dg_qu","br":"dg_br","ow":"dg_ow",
  "fo":"dg_fo","ju":"dg_ju","ov":"dg_ov","er":"dg_er","la":"dg_la",
};
const TRIGRAPH_MAP = {
  "the":"tg_the","bro":"tg_bro","own":"tg_own","ove":"tg_ove","ver":"tg_ver",
};
const KEY_DWELL_KEYS = ["e","o","t","h","r","u","space"];

const freshState = () => ({
  keyDownTimes:  {},
  dwellTimes:    [],
  keyDwells:     {},
  flightTimes:   [],
  allIntervals:  [],
  lastKeyUpTime: null,
  keyCount:      0,
  backspaceCount:0,
  startTime:     null,
  dgTimes:       {},
  tgTimes:       {},
  keySeq:        [],
});

export default function App() {
  const [inputVal,   setInputVal]   = useState("");
  const [complete,   setComplete]   = useState(false);
  const [errorFlash, setErrorFlash] = useState(false);
  const [metrics,    setMetrics]    = useState({ dwell:0,flight:0,wpm:0,entropy:0,count:0 });
  const [status,  setStatus]  = useState({ text:"Waiting for input...", type:"" });
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);

  const ks          = useRef(freshState());
  const textareaRef = useRef(null);
  const progress    = Math.min(100,(inputVal.length/TARGET.length)*100);

  const refreshMetrics = useCallback(() => {
    const s=ks.current;
    if (!s.startTime) return;
    const elapsed=(performance.now()-s.startTime)/1000;
    const words=inputVal.trim().split(/\s+/).filter(Boolean).length;
    setMetrics({
      dwell:  Math.round(avg(s.dwellTimes)),
      flight: Math.round(avg(s.flightTimes)),
      wpm:    Math.round(words/elapsed*60),
      entropy:entropy([...s.dwellTimes,...s.flightTimes]).toFixed(2),
      count:  s.keyCount,
    });
  },[inputVal]);

  const handleKeyDown = useCallback((e) => {
    const now=performance.now();  // ms ✓
    const s=ks.current;
    if (!s.startTime) s.startTime=now;
    s.keyDownTimes[e.code]=now;

    if (s.lastKeyUpTime!==null) {
      const flight=now-s.lastKeyUpTime;
      s.flightTimes.push(flight);
      s.allIntervals.push(flight);
    }

    if (e.key==="Backspace") s.backspaceCount++;
    s.keyCount++;

    // Key sequence for digraph/trigraph
    const k = e.key===" "?"space":e.key.toLowerCase();
    s.keySeq.push({key:k, time:now});

    // Digraph — last 2 keys
    if (s.keySeq.length>=2) {
      const p=s.keySeq[s.keySeq.length-2];
      const c=s.keySeq[s.keySeq.length-1];
      const pair=p.key+c.key;
      const feat=DIGRAPH_MAP[pair];
      if (feat) {
        if (!s.dgTimes[feat]) s.dgTimes[feat]=[];
        const gap=c.time-p.time;
        if (gap>=20&&gap<=1500) s.dgTimes[feat].push(gap);
      }
      // Space digraph
      if (c.key==="space") {
        if (!s.dgTimes["dg_sp"]) s.dgTimes["dg_sp"]=[];
        const gap=c.time-p.time;
        if (gap>=20&&gap<=1500) s.dgTimes["dg_sp"].push(gap);
      }
    }

    // Trigraph — last 3 keys
    if (s.keySeq.length>=3) {
      const k1=s.keySeq[s.keySeq.length-3];
      const k2=s.keySeq[s.keySeq.length-2];
      const k3=s.keySeq[s.keySeq.length-1];
      const tri=k1.key+k2.key+k3.key;
      const feat=TRIGRAPH_MAP[tri];
      if (feat) {
        if (!s.tgTimes[feat]) s.tgTimes[feat]=[];
        const span=k3.time-k1.time;
        if (span>=30&&span<=3000) s.tgTimes[feat].push(span);
      }
    }

    refreshMetrics();
  },[refreshMetrics]);

  const handleKeyUp = useCallback((e) => {
    const now=performance.now();  // ms ✓
    const s=ks.current;

    if (s.keyDownTimes[e.code]!==undefined) {
      const dwell=now-s.keyDownTimes[e.code];
      if (dwell>=20&&dwell<=800) {
        s.dwellTimes.push(dwell);
        const k=e.key===" "?"space":e.key.toLowerCase();
        if (KEY_DWELL_KEYS.includes(k)) {
          if (!s.keyDwells[k]) s.keyDwells[k]=[];
          s.keyDwells[k].push(dwell);
        }
      }
      delete s.keyDownTimes[e.code];
    }
    s.lastKeyUpTime=now;
    refreshMetrics();
  },[refreshMetrics]);

  const handleChange = useCallback((e) => {
    const val=e.target.value;
    setInputVal(val);
    if (val===TARGET) {
      setComplete(true);
      setStatus({text:"✓ Complete. Press IDENTIFY USER.",type:"ok"});
    } else {
      setComplete(false);
      if (val!==""&&!TARGET.startsWith(val)) {
        setErrorFlash(true);
        setTimeout(()=>setErrorFlash(false),300);
      }
      if (status.type==="ok") setStatus({text:"Waiting for input...",type:""});
    }
  },[status.type]);

  // ── Build 24 features (all ms) ──
  const buildFeatures = () => {
    const s=ks.current;
    const elapsedSec=(performance.now()-s.startTime)/1000;
    const words=inputVal.trim().split(/\s+/).filter(Boolean).length;
    const dw=s.dwellTimes, fl=s.flightTimes;
    const dwell_mean=avg(dw), flight_mean=avg(fl);
    const char_count=s.keyCount-s.backspaceCount;

    const dg=(feat)=>avg(s.dgTimes[feat]||[]);
    const tg=(feat)=>avg(s.tgTimes[feat]||[]);
    const kd=(key) =>avg(s.keyDwells[key]||[]);

    return {
      dwell_mean:     +dwell_mean.toFixed(2),
      dwell_std:      +std(dw).toFixed(2),
      dwell_cv:       +cv(dw).toFixed(4),
      flight_mean:    +flight_mean.toFixed(2),
      flight_std:     +std(fl).toFixed(2),
      flight_cv:      +cv(fl).toFixed(4),
      timing_entropy: +entropy([...dw,...fl]).toFixed(4),
      total_duration: +(elapsedSec*1000).toFixed(2),
      wpm:            +(words/elapsedSec*60).toFixed(2),
      dg_th: +dg("dg_th").toFixed(2), dg_he: +dg("dg_he").toFixed(2),
      dg_qu: +dg("dg_qu").toFixed(2), dg_br: +dg("dg_br").toFixed(2),
      dg_ow: +dg("dg_ow").toFixed(2), dg_fo: +dg("dg_fo").toFixed(2),
      dg_ju: +dg("dg_ju").toFixed(2), dg_ov: +dg("dg_ov").toFixed(2),
      dg_er: +dg("dg_er").toFixed(2), dg_la: +dg("dg_la").toFixed(2),
      dg_sp: +dg("dg_sp").toFixed(2),
      tg_the: +tg("tg_the").toFixed(2), tg_bro: +tg("tg_bro").toFixed(2),
      tg_own: +tg("tg_own").toFixed(2), tg_ove: +tg("tg_ove").toFixed(2),
      tg_ver: +tg("tg_ver").toFixed(2),
      kd_e: +kd("e").toFixed(2),  kd_o: +kd("o").toFixed(2),
      kd_t: +kd("t").toFixed(2),  kd_h: +kd("h").toFixed(2),
      kd_r: +kd("r").toFixed(2),  kd_u: +kd("u").toFixed(2),
      kd_space: +kd("space").toFixed(2),
      backspace_rate: s.keyCount>0?+(s.backspaceCount/s.keyCount).toFixed(4):0,
    };
  };

  const submitPrediction = async () => {
    if (!complete) return;
    setLoading(true);
    setStatus({text:"Analyzing keystroke pattern...",type:"loading"});
    try {
      const res=await fetch(`${API_BASE}/predict`,{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify(buildFeatures()),
      });
      const data=await res.json();
      if (!res.ok) throw new Error(data.error||"Server error");
      setResult(data);
      setStatus({text:"✓ Identification complete.",type:"ok"});
    } catch(err) {
      setStatus({text:"✗ Error: "+err.message,type:"error"});
    } finally { setLoading(false); }
  };

  const resetAll = () => {
    ks.current=freshState();
    setInputVal(""); setComplete(false); setErrorFlash(false);
    setMetrics({dwell:0,flight:0,wpm:0,entropy:0,count:0});
    setStatus({text:"Waiting for input...",type:""});
    setLoading(false); setResult(null);
    setTimeout(()=>textareaRef.current?.focus(),50);
  };

  useEffect(()=>{ textareaRef.current?.focus(); },[]);

  const inputClass=["typing-input",complete?"complete":"",errorFlash?"error-flash":""].filter(Boolean).join(" ");
  const statusClass=["status-bar",status.type].filter(Boolean).join(" ");
  return (
    <div className="app">

      <div className="card">
        <div className="section-label">▸ 01 / TYPE THE SENTENCE BELOW (all lowercase)</div>
        <div className="sentence-display">{TARGET}</div>

        <textarea
          ref={textareaRef}
          className={inputClass}
          rows={2}
          value={inputVal}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onKeyUp={handleKeyUp}
          autoComplete="off" autoCorrect="off"
          autoCapitalize="off" spellCheck={false}
          placeholder="Start typing here..."
        />
        <div className="progress-bar-wrap">
          <div className="progress-bar-fill" style={{width:`${progress}%`}}/>
        </div>

        <div style={{marginTop:18}}>
          <div className="section-label">▸ 02 / REAL-TIME BIOMETRICS</div>
          <div className="metrics-grid">
            <MetricBox val={`${metrics.dwell}ms`}  lbl="Dwell Mean"  />
            <MetricBox val={`${metrics.flight}ms`} lbl="Flight Mean" />
            <MetricBox val={metrics.wpm}            lbl="WPM"         />
            <MetricBox val={metrics.entropy}        lbl="Entropy"     />
            <MetricBox val={metrics.count}          lbl="Key Count"   />
          </div>
        </div>

        <div>
          <button className="btn" disabled={!complete||loading} onClick={submitPrediction}>
            {loading?<><span className="spinner"/>ANALYZING...</>:"◈ IDENTIFY USER"}
          </button>
          <button className="btn reset" onClick={resetAll}>↺ RESET</button>
        </div>
        <div className={statusClass}>
          {status.type==="loading"&&<span className="spinner"/>}
          {status.text}
        </div>
      </div>

      {result&&<ResultPanel result={result}/>}
    </div>
  );
}

function ResultPanel({result}) {
  const maxA=result.max_appearances;
  return (
    <div className="card result-panel">
      <div className="section-label">▸ 03 / IDENTIFICATION RESULT</div>
      <div className="winner-box">
        <div className="winner-label">IDENTIFIED USER</div>
        <div className="winner-name">{result.winner.toUpperCase()}</div>
        <div className="winner-votes">
          appeared in {result.winner_appearances} of {maxA} slots
          &nbsp;·&nbsp; score: {result.weighted_scores[result.winner]}
        </div>
      </div>
      <div className="voting-explainer">
        <div className="explainer-title">HOW THE WINNER WAS CHOSEN</div>
        <div className="explainer-body">
          Each of the {result.total_models} models independently predicted its own Top-3
          using different features and different training data.
          The user appearing most across all {maxA} slots wins.
        </div>
      </div>
      <div className="section-label" style={{marginTop:24,marginBottom:8}}>
        EACH MODEL'S INDEPENDENT TOP-3
      </div>
      <div className="model-grid">
        {Object.entries(result.per_model_top3).map(([mname,top3])=>(
          <div className="model-card" key={mname}>
            <div className="model-name">{mname}</div>
            {top3.map((item,i)=>(
              <div className={`rank-row ${item.user===result.winner?"rank-row-winner":""}`} key={i}>
                <span className="rank-num">#{i+1}</span>
                <span className="rank-user">{item.user}</span>
                <span className="rank-conf">{item.confidence}%</span>
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="section-label" style={{marginTop:24,marginBottom:8}}>
        VOTE COUNT
      </div>
      {Object.entries(result.appearance_counts).map(([user,count])=>(
        <div className="vote-row" key={user}>
          <div className={`vote-name ${user===result.winner?"vote-name-winner":""}`}>
            {user===result.winner?"★ ":""}{user}
          </div>
          <div className="vote-bar-wrap">
            <div className="vote-bar-fill" style={{width:`${(count/maxA)*100}%`}}/>
          </div>
          <div className="vote-count-label">{count}/{maxA}</div>
        </div>
      ))}
    </div>
  );
}

function MetricBox({val,lbl}) {
  return (
    <div className="metric-box">
      <div className="metric-val">{val}</div>
      <div className="metric-lbl">{lbl}</div>
    </div>
  );
}