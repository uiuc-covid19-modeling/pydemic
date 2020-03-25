/*
 *
 *  $ node server_neherlab.js
 *
 */

var fs = require('fs');
var text = fs.readFileSync("run.js").toString();
eval(text);

exports.wrapper().then(function(result) {

  var times = [];
  var susceptible = [];
  var exposed = [];
  var infectious = [];
  var recovered = [];
  var hospitalized = [];
  var critical = [];
  var overflow = [];
  var discharged = [];
  var intensive = [];
  var dead = [];
  var traj = result.deterministicTrajectory

  for (var i in traj) {
    times.push(traj[i].time);
    susceptible.push(traj[i].susceptible.total);
    exposed.push(traj[i].exposed.total);
    infectious.push(traj[i].infectious.total);
    recovered.push(traj[i].recovered.total);
    hospitalized.push(traj[i].hospitalized.total);
    critical.push(traj[i].critical.total);
    overflow.push(traj[i].overflow.total);
    discharged.push(traj[i].discharged.total);
    intensive.push(traj[i].intensive.total);
    dead.push(traj[i].dead.total);
  }

  var data = 
  {
    "times": times, 
    "suspectible": susceptible, 
    "exposed": exposed, 
    "infectious": infectious, 
    "recovered": recovered, 
    "hospitalized": hospitalized, 
    "critical": critical, 
    "overflow": overflow, 
    "discharged": discharged, 
    "intensive": intensive, 
    "dead": dead
  };
  var text = JSON.stringify(data);

  console.log(text);
});


// set up ad hoc webserver
var http = require('http');

http.createServer(function (req, res) {
  res.writeHead(200, {'Content-Type': 'text/plain'});
  res.end('Hello World!');
}).listen(8080);




