/*
 *
 *  $ node server_neherlab.js
 *
 */




var runFilename = "run.js";

/*
import { run } from './run';
 */

var fs = require('fs');
var text = fs.readFileSync(runFilename).toString();
eval(text);

console.log("got run as exports.default");

console.log(exports.wrapper());

/*
var text = fs.readFileSync("types/Param.types.js").toString();
eval(text);

var params = new AllParamsFlat();
 */


/*
var runFunction = exports.default;
console.log(runFunction);



// AllParamsFlat
var params;

// SeverityTableRow[]
var severity;

// OneCountryAgeDistribution
var ageDistribution;

// TimeSeries
var containment;

runFunction(params, severity, ageDistribution, containment);
 */



/*
 */


/*


eval(fs.readFileSync(runFilename).toString());
 */


// load neherlab files
//var filenameMainjs = "neherlab/main.js";
//var filenameHomejs = "neherlab/Home.js";
/*
var fs = require('fs');
eval(fs.readFileSync(filenameHomejs).toString());
 */

//var Home = require("neherlab/Home.js");


/*
var tools = require('./examples/tools.js');
console.log(tools.zemba);
console.log(tools.foo);
tools.foo();
 */

/*
var tools = require('./examples/tools.js');
//eval(tools.readFileSync(tools));

//console.log(fs.readFileSync('./examples/tools.js').toString());

eval(fs.readFileSync('./examples/tools.js').toString());

//console.log(fs.readFileSync(tools).toString());
//console.log(tools.zemba);
//console.log(tools.foo);

zemba();
foo();
 */



// set up ad hoc webserver
var http = require('http');

http.createServer(function (req, res) {
  res.writeHead(200, {'Content-Type': 'text/plain'});
  res.end('Hello World!');
}).listen(8080);





