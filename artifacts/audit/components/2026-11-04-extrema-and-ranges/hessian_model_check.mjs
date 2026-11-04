// Independent exact-arithmetic model of the LDL acceptance criterion.
// Run: node hessian_model_check.mjs
// This does NOT execute FrankenSymPy's Rust verifier or replace its tests.
// Reviewed implementation blob: bc28651454cf85ec82a01caf3c639991d6f29012.
function gcd(a,b) {
  a=a<0n?-a:a; b=b<0n?-b:b;
  while(b) [a,b]=[b,a%b];
  return a;
}
function q(a,b=1n) {
  a=BigInt(a); b=BigInt(b);
  if(b<0n) { a=-a; b=-b; }
  if(!b) throw Error("zero denominator");
  const g=gcd(a,b);
  return [a/g,b/g];
}
const z=q(0);
const cmp=(a,b)=>a[0]*b[1]<b[0]*a[1]?-1:a[0]*b[1]>b[0]*a[1]?1:0;
const add=(a,b)=>q(a[0]*b[1]+b[0]*a[1],a[1]*b[1]);
const neg=a=>[-a[0],a[1]];
const sub=(a,b)=>add(a,neg(b));
const mul=(a,b)=>q(a[0]*b[0],a[1]*b[1]);
const div=(a,b)=>q(a[0]*b[1],a[1]*b[0]);
const iz=()=>[z,z], point=a=>[q(a),q(a)];
const isub=(a,b)=>[sub(a[0],b[1]),sub(a[1],b[0])];
function imin(xs) { return xs.reduce((a,b)=>cmp(a,b)<0?a:b); }
function imax(xs) { return xs.reduce((a,b)=>cmp(a,b)>0?a:b); }
function imul(a,b) {
  const p=[mul(a[0],b[0]),mul(a[0],b[1]),mul(a[1],b[0]),mul(a[1],b[1])];
  return [imin(p),imax(p)];
}
function isquare(a) {
  if(cmp(a[0],z)>=0) return [mul(a[0],a[0]),mul(a[1],a[1])];
  if(cmp(a[1],z)<=0) return [mul(a[1],a[1]),mul(a[0],a[0])];
  return [z,imax([mul(a[0],a[0]),mul(a[1],a[1])])];
}
function idiv(a,b) {
  if(cmp(b[0],z)<=0&&cmp(b[1],z)>=0) throw Error("zero interval");
  return imul(a,[div(q(1),b[1]),div(q(1),b[0])]);
}
// Interval LDL model. This retains zero-reaching pivots only when their
// complete remaining Schur column is exactly zero.
function ldl(h) {
  const n=h.length;
  const l=Array.from({length:n},()=>Array.from({length:n},iz)),d=[];
  let pd=true;
  for(let i=0;i<n;i++) {
    let diag=h[i][i];
    for(let k=0;k<i;k++) diag=isub(diag,imul(isquare(l[i][k]),d[k]));
    if(cmp(diag[0],z)<0) return {psd:false,pd:false};
    const positive=cmp(diag[0],z)>0;
    pd&&=positive;
    for(let j=i+1;j<n;j++) {
      let e=h[j][i];
      for(let k=0;k<i;k++) e=isub(e,imul(imul(l[j][k],l[i][k]),d[k]));
      if(positive) l[j][i]=idiv(e,diag);
      else if(cmp(e[0],z)!==0||cmp(e[1],z)!==0) return {psd:false,pd:false};
    }
    d.push(diag);
  }
  return {psd:true,pd};
}
// Independent oracle: all principal minors characterize PSD. Leading
// principal minors characterize PD. Neither oracle uses elimination.
function criterion2(a,b,c) {
  return {psd:a>=0n&&c>=0n&&a*c-b*b>=0n,pd:a>0n&&a*c-b*b>0n};
}
function criterion3(a,b,c,d,e,f) {
  const det=a*d*f+2n*b*c*e-a*e*e-d*c*c-f*b*b;
  return {
    psd:a>=0n&&d>=0n&&f>=0n&&a*d-b*b>=0n&&a*f-c*c>=0n&&d*f-e*e>=0n&&det>=0n,
    pd:a>0n&&a*d-b*b>0n&&det>0n,
  };
}
let exact2=0,psd2=0,exact3=0,psd3=0,interval2=0,intervalAccepted=0,conservativeRefusals=0;
for(let a=-5;a<=5;a++) for(let b=-5;b<=5;b++) for(let c=-5;c<=5;c++) {
  const r=ldl([[point(a),point(b)],[point(b),point(c)]]);
  const t=criterion2(BigInt(a),BigInt(b),BigInt(c));
  if(r.psd!==t.psd||(r.psd&&r.pd!==t.pd)) throw Error("2x2 mismatch "+[a,b,c]);
  exact2++; psd2+=r.psd;
}
for(let a=-2;a<=2;a++) for(let b=-2;b<=2;b++) for(let c=-2;c<=2;c++)
for(let d=-2;d<=2;d++) for(let e=-2;e<=2;e++) for(let f=-2;f<=2;f++) {
  const r=ldl([[point(a),point(b),point(c)],[point(b),point(d),point(e)],[point(c),point(e),point(f)]]);
  const t=criterion3(...[a,b,c,d,e,f].map(BigInt));
  if(r.psd!==t.psd||(r.psd&&r.pd!==t.pd)) throw Error("3x3 mismatch "+[a,b,c,d,e,f]);
  exact3++; psd3+=r.psd;
}
const intervals=[];
for(let lo=-2;lo<=2;lo++) for(let hi=lo;hi<=2;hi++) intervals.push([BigInt(lo),BigInt(hi)]);
for(const a of intervals) for(const b of intervals) for(const c of intervals) {
  const aq=a.map(x=>q(x)),bq=b.map(x=>q(x)),cq=c.map(x=>q(x));
  const r=ldl([[aq,bq],[bq,cq]]);
  let allPSD=true,allPD=true;
  // The symmetric interval matrix is a convex hull of these vertices.
  // PSD/PD are convex cones, so vertex membership checks the entire box.
  for(const av of a) for(const bv of b) for(const cv of c) {
    const t=criterion2(av,bv,cv);
    allPSD&&=t.psd; allPD&&=t.pd;
  }
  if(r.psd&&!allPSD) throw Error("interval false PSD");
  if(r.pd&&!allPD) throw Error("interval false PD");
  interval2++; intervalAccepted+=r.psd; conservativeRefusals+=allPSD&&!r.psd;
}
console.log(JSON.stringify({
  scope:"Independent exact-BigInt rational model compared to principal minors; not Rust execution",
  exact2,psd2,exact3,psd3,interval2,intervalAccepted,conservativeRefusals,
  unsound_acceptances:0,
},null,2));
