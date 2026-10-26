
(()=>{
  const root=document.getElementById("mjfit-construction-view");
  const svg=root.querySelector(".construction-svg");
  const data=JSON.parse(root.querySelector("[data-geometry]").textContent);
  const detail=root.querySelector("[data-detail]");
  const buttons=[...root.querySelectorAll("[data-mode]")];
  const modes=["section","prism","exploded"];
  let mode="section";
  const initial=window.openai?.widgetState?.modelContent?.mode;
  if(modes.includes(initial))mode=initial;
  function element(tag,attributes,text){
    const e=document.createElementNS("http://www.w3.org/2000/svg",tag);
    for(const [key,value]of Object.entries(attributes))e.setAttribute(key,String(value));
    if(text!==undefined)e.textContent=text;
    return e;
  }
  function draw(){
    const width=Math.max(260,root.getBoundingClientRect().width);
    const narrow=width<540;
    const height=narrow?500:360;
    const scale=narrow?(width-46)/96:Math.min(5,(width-40)/145);
    svg.setAttribute("viewBox",`0 0 ${width} ${height}`);
    svg.setAttribute("height",height);
    svg.replaceChildren();
    svg.append(element("title",{},`星形孔构造：${mode==="section"?"二维截面":mode==="prism"?"挤出凸棱柱":"分块示意"}。夹具和插销各10个凸块，分别属于同一个刚体。`));
    buttons.forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.mode===mode)));
    const names={plate:"夹具 · 10 个凸块",peg:"插销 · 10 个凸块"};
    for(const [kind,series]of [["plate",1],["peg",2]]){
      const origin=kind==="plate"
        ?[narrow?width/2:width*0.31,narrow?177:220]
        :[narrow?width/2:width*0.79,narrow?417:220];
      const titleY=narrow?(kind==="plate"?28:284):28;
      svg.append(element("text",{x:origin[0],y:titleY,"text-anchor":"middle"},names[kind]));
      const pieces=[...data.groups[kind]];
      const scene=[];
      for(const piece of pieces){
        const polygon=piece.polygon_mm;
        const center=polygon.reduce((sum,p)=>[sum[0]+p[0]/polygon.length,sum[1]+p[1]/polygon.length],[0,0]);
        const length=Math.hypot(...center)||1;
        const distance=mode==="exploded"?(kind==="plate"?7:5):0;
        const dx=distance*center[0]/length,dy=distance*center[1]/length;
        const project=(p,z)=>{
          const x=p[0]+dx,y=p[1]+dy;
          return mode==="section"
            ?[origin[0]+x*scale,origin[1]-y*scale]
            :[origin[0]+(x-y)*0.76*scale,origin[1]+(x+y)*0.33*scale-z*0.9*scale];
        };
        const z=piece.z1_mm-piece.z0_mm;
        const top=polygon.map(p=>project(p,z));
        const bottom=polygon.map(p=>project(p,0));
        if(mode!=="section"){
          for(let i=0;i<polygon.length;i++){
            const j=(i+1)%polygon.length;
            const points=[bottom[i],bottom[j],top[j],top[i]];
            scene.push({points,color:series,opacity:0.18,depth:(bottom[i][1]+bottom[j][1])/2,kind:"side",name:piece.name});
          }
        }
        scene.push({points:top,color:series,opacity:0.38,depth:top.reduce((s,p)=>s+p[1],0)/top.length,kind:"top",name:piece.name});
      }
      scene.sort((a,b)=>a.depth-b.depth);
      for(const face of scene){
        const mark=element("polygon",{
          points:face.points.map(p=>p.map(v=>v.toFixed(2)).join(",")).join(" "),
          fill:`var(--viz-series-${face.color})`,"fill-opacity":face.opacity,
          stroke:"var(--border)","stroke-width":1,
        });
        mark.append(element("title",{},face.name));
        svg.append(mark);
      }
      if(mode==="section"){
        const outline=kind==="plate"?data.hole_mm:data.peg_outline_mm;
        svg.append(element("polygon",{
          points:outline.map(p=>`${origin[0]+p[0]*scale},${origin[1]-p[1]*scale}`).join(" "),
          fill:"none",stroke:"var(--foreground)","stroke-width":1.5
        }));
      }
    }
    detail.textContent=mode==="exploded"
      ?"为观察分块而分离；实际每组凸块共用一个刚体。"
      :mode==="prism"
      ?"夹具厚 8 mm · 销长 20 mm · 每个凸多边形沿 z 挤出"
      :"夹具 10 块 · 插销 10 块 · 2 个刚体 · 单边间隙 0.5 mm";
  }
  buttons.forEach(button=>button.addEventListener("click",()=>{
    mode=button.dataset.mode;draw();
    if(window.openai?.setWidgetState)window.openai.setWidgetState({modelContent:{mode},privateContent:null}).catch(()=>{});
  }));
  window.addEventListener("openai:set_globals",event=>{
    const next=event.detail?.globals?.widgetState?.modelContent?.mode;
    if(modes.includes(next)){mode=next;draw();}
  });
  new ResizeObserver(draw).observe(root);
  draw();
})();
