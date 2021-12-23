# get incomes
curl http://localhost:5000/duts

# add new income
curl -X POST -H "Content-Type: application/json" -d '{
  "description": "another test device",
  "name": "iphone 188",
  "_id": "54321"
}' http://localhost:5000/duts

# check if lottery was added
curl localhost:5000/duts

# remove an income
curl -X DELETE -H "Content-Type: application/json" -d '{
  "description": "another test device",
  "name": "iphone 188",
  "_id": "54321"
}' http://localhost:5000/duts

# check if lottery was added
curl localhost:5000/duts