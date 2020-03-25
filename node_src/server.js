/*
 *
 *  $ node server_neherlab.js
 *
 */

var fs = require('fs');
var text = fs.readFileSync("wrapper.js").toString();
eval(text);

// set up ad hoc webserver
var http = require('http');
const url = require('url');
const querystring = require('querystring');

http.createServer(function (req, res) {

  if (req.method === 'POST') {

    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {

      var argdata = JSON.parse(body);

      console.log(argdata);

      exports.wrapper(argdata).then(function(result) {

        console.log(result);

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

          console.log(traj[i].time);
          console.log('susceptible');
          console.log(traj[i].susceptible);
          //console.log(traj[i].exposed);
          console.log('infectious');
          console.log(traj[i].infectious);
          console.log('recovered');
          console.log(traj[i].recovered);
          console.log('hospitalized');
          console.log(traj[i].hospitalized);
          //console.log(traj[i].critical);
          //console.log(traj[i].overflow);
          //console.log(traj[i].discharged);
          //console.log(traj[i].intensive);
          //console.log(traj[i].dead);
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

        res.setHeader('Content-Type', 'application/json');
        res.end(text);

      });

    })

  }

}).listen(8081);




