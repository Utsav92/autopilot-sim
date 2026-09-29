require('http').createServer((q,r)=>{let b='';q.on('data',c=>b+=c);q.on('end',()=>{const j=JSON.parse(b);
 r.writeHead(200,{'content-type':'application/json'});
 r.end(JSON.stringify({content:[{type:'text',text:'Here you go:\n{"weather":"fog","density":60,"events":{"cutin":true,"bogus":true},"setSpeedKmh":300,"tags":["dense fog","cut-in"]}'}],_seen:{key:q.headers['x-api-key'],model:j.model}}));});}).listen(8899);
