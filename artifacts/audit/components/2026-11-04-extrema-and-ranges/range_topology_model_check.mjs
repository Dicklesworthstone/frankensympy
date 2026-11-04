// Independent domain-cell/intersection/union model.
// Run: node range_topology_model_check.mjs
// This does NOT execute FrankenSymPy Python/native code or replace its tests.
// Reviewed implementation blob: a13335f95dc271776ecb8aa53034ff561e5eedf6.
// All finite coordinates are exact binary integers/halves/quarters; no
// rounded numerical value or approximate comparison enters this model.
function cmp(a,b) { return a<b?-1:a>b?1:0; }
function pieces(n,mask) {
  const out=[];
  let p=0,last=2*n;
  while(p<=last) {
    if(!(mask&(1<<p))) { p++; continue; }
    const first=p;
    while(p+1<=last&&(mask&(1<<(p+1)))) p++;
    const end=p;
    out.push([
      first===0?-Infinity:Math.floor((first-1)/2),
      end===2*n?Infinity:Math.floor(end/2),
      first%2===0,end%2===0,
    ]);
    p++;
  }
  return out;
}
function intersect(l,r) {
  let o=cmp(l[0],r[0]);
  const lo=o>=0?l[0]:r[0],lop=o>0?l[2]:o<0?r[2]:l[2]||r[2];
  o=cmp(l[1],r[1]);
  const hi=o<=0?l[1]:r[1],hip=o<0?l[3]:o>0?r[3]:l[3]||r[3];
  if(lo>hi||lo===hi&&(!Number.isFinite(lo)||lop||hip)) return null;
  return [lo,hi,lop,hip];
}
function merge(parts) {
  parts=parts.slice().sort((l,r)=>cmp(l[0],r[0])||cmp(l[2],r[2]));
  const m=[];
  for(const p of parts) {
    if(!m.length) { m.push(p); continue; }
    const v=m[m.length-1],o=cmp(p[0],v[1]);
    if(o>0||o===0&&p[2]&&v[3]) { m.push(p); continue; }
    const e=cmp(p[1],v[1]);
    m[m.length-1]=[v[0],e>0?p[1]:v[1],v[2],e>0?p[3]:e<0?v[3]:p[3]&&v[3]];
  }
  return m;
}
// Independent semantics: classify each probe directly in its chart cell
// and test its original domain endpoints, without constructing components.
function contains(p,x) {
  return (p[0]<x||p[0]===x&&!p[2])&&(p[1]>x||p[1]===x&&!p[3]);
}
function cell(n,x) {
  for(let i=0;i<n;i++) {
    if(x<i) return 2*i;
    if(x===i) return 2*i+1;
  }
  return 2*n;
}
let chartMasks=0,clippedDomains=0,domainMemberships=0;
for(let n=0;n<=4;n++) {
  const bounds=[-Infinity,...Array.from({length:2*n+5},(_,i)=>(i-2)/2),Infinity];
  const probes=[-1000,...Array.from({length:4*n+13},(_,i)=>(i-4)/4),1000];
  for(let mask=0;mask<(1<<(2*n+1));mask++) {
    const ps=pieces(n,mask);
    chartMasks++;
    for(let a=0;a<bounds.length;a++) for(let b=a;b<bounds.length;b++)
    for(let flags=0;flags<4;flags++) {
      const d=[
        bounds[a],bounds[b],
        !!(flags&1)||bounds[a]===-Infinity,
        !!(flags&2)||bounds[b]===Infinity,
      ];
      const actual=ps.map(p=>intersect(p,d)).filter(Boolean);
      clippedDomains++;
      for(const x of probes) {
        const expected=contains(d,x)&&!!(mask&(1<<cell(n,x)));
        if(actual.some(p=>contains(p,x))!==expected)
          throw Error("coverage mismatch "+JSON.stringify({n,mask,d,x}));
        domainMemberships++;
      }
    }
  }
}
const pvals=[-Infinity,-2,-1,0,1,2,Infinity],all=[];
for(let i=0;i<pvals.length;i++) for(let j=i;j<pvals.length;j++) for(let f=0;f<4;f++) {
  const p=[pvals[i],pvals[j],!!(f&1)||pvals[i]===-Infinity,!!(f&2)||pvals[j]===Infinity];
  if(intersect(p,p)) all.push(p);
}
let pairUnions=0,mergeMemberships=0;
const probes=[-100,...Array.from({length:25},(_,i)=>(i-12)/4),100];
for(const a of all) for(const b of all) {
  const m=merge([a,b]);
  pairUnions++;
  for(const x of probes) {
    if(m.some(p=>contains(p,x))!==(contains(a,x)||contains(b,x)))
      throw Error("merge mismatch "+JSON.stringify({a,b,x}));
    mergeMemberships++;
  }
}
console.log(JSON.stringify({
  scope:"Independent interval-cell and union model checked against direct membership; not Python/native execution",
  chartMasks,clippedDomains,domainMemberships,pairUnions,mergeMemberships,mismatches:0,
},null,2));
