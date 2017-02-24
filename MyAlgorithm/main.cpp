#include <limits>
#include <set>
#include <map>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <mpi.h>
#include <lemon/concepts/digraph.h>
#include <lemon/smart_graph.h>
#include <lemon/list_graph.h>
#include <lemon/lgf_reader.h>
#include <lemon/dijkstra.h>

using namespace lemon;
using namespace std;


using namespace lemon;

typedef unsigned long long timestamp_t;

//Used to calculate the time a process needed to terminate.
static timestamp_t
get_timestamp ()
{
	struct timeval now;
	gettimeofday (&now, NULL);
	return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

//Structure used for message transference in MPI.
struct taskMessage
{
    int length;
    int distance;
};

//Edge storage in order to restore the graph
struct edgeStorage
{
    ListDigraphBase::Arc arc;
    ListDigraphBase::Node target;
    int weight;
};

/**
 * Insert in map method is used to add a new path with a distance to a multimap avoiding duplicates.
 */
void insertInMap(multimap<int, vector<int>> *shortestPaths, taskMessage buffer, vector<int> newPath)
{
    multimap<int,vector<int>>::const_iterator it  = shortestPaths->lower_bound(buffer.distance);
    multimap<int,vector<int>>::const_iterator it2 = shortestPaths->upper_bound(buffer.distance);

    bool isEqual = false;
    while (it !=it2 )
    {
        if(equal(it->second.begin(),it->second.end(),newPath.begin()))
        {
            isEqual = true;
        }
        ++it;
    }
    if(!isEqual)
    {
        shortestPaths->insert(make_pair(buffer.distance,newPath));
    }
}

//Checks if subGroup is in group.
bool in(vector<int> const &group, vector<int> const &subGroup)
{
    size_t const     subSize = subGroup.size();
    int                   i = 0;

    while (i < subSize && find(group.begin(), group.end(), subGroup[i]) != group.end())
    {
        i++;
    }
    return (i == subSize);
}

//mpic++ main.cpp -o main Graph.cpp YenTopKShortestPathsAlg.cpp DijkstraShortestPathAlg.cpp -lemon
//mpirun -np 4 ./main
void run(int *argc, char ***argv  )
{
    //Initiate the program run time counter.
    timestamp_t t00 = get_timestamp();

    //Here starts the MPI block.
    MPI_Init(argc, argv);

    //Init the k for the number of paths
    int k = 10;
    //Init the rank of the MPI process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,	&rank);
    //Init the number of processes.
    int nOSlaves;
    MPI_Comm_size(MPI_COMM_WORLD,	&nOSlaves);

    //Define the Parameter types of the lemon libraries.
    typedef lemon::ListDigraph Graph;
    typedef Graph::Arc EdgeIt;
    typedef Graph::Arc Edge;
    typedef Graph::NodeIt NodeIt;
    typedef Graph::Node Node;
    typedef Graph::ArcMap<int> LengthMap;
    using lemon::INVALID;

    //Initialize the test graph.
    Graph g;

    for(int i = 1; i<= 1070376; i++)
    {
        g.addNode();
    }

    LengthMap length(g);

    string line;
    ifstream file ("data/florida.gr");

    if (file.is_open())
    {
        string word;
        while (getline (file,line) )
        {
            std::string::size_type sz;
            file >> word;
            int node1 = stoi(word, &sz);
            file >> word;
            int node2 = stoi(word, &sz);
            file >> word;
            double distance = stod(word, &sz);
            Edge edge = g.addArc(g.nodeFromId(node1),g.nodeFromId(node2));
            length[edge] = distance;
        }
        file.close();
    }
    else cout << "Unable to open file" << endl;

    Node s = g.nodeFromId(15);
    Node t = g.nodeFromId(300);

    cout << "done adding graph, starting execution" << endl;

    //If the process is a coordinator.
    if(rank==0)
    {
        //Contains the shortest paths.
        multimap<int,vector<int>> shortestPaths;
        //Contains the already tried examples **Debug purpose**
        vector<vector<int>> removedPaths;
        //Contains the job of the currently running slaves at their index.
        vector<int> runningTasks[nOSlaves];

        //Create a set which orders the elements according to their size.
        auto comp = [](const vector<int>& a, const vector<int>& b) -> bool
        {
            if (a.size() < b.size())
                return true;
            if (a.size() > b.size())
                return false;
            return a < b;
        };
        auto path = std::set <vector<int>, decltype(comp)> (comp);

        timestamp_t td1 = get_timestamp();
        //Create an instance of Dijkstra's algorithm and execute it.
        Dijkstra<Graph, LengthMap> dijkstra_test(g,length);
        dijkstra_test.run(s);

        vector<int> shortestPath;        //Print the path the algorithm found.
        for (Node v=t;v != s; v=dijkstra_test.predNode(v))
        {
            shortestPath.push_back(g.id(v));
            //std::cout << g.id(v) << "<-";
        }
        shortestPath.push_back(g.id(s));
        //cout << g.id(s) << endl;;
        timestamp_t td2 = get_timestamp();

        long double secs = (td2 - td1) / 1000000.0L;

        cout << secs << endl;

        //Fill the path set with all edges between source s and sink t. These will be used as tasks for the slaves.
        ListDigraphBase::Arc v=dijkstra_test.predArc(t);
        path.insert(vector<int>{g.id(v)});
        //cout << g.id(v);
        do
        {
            path.insert(vector<int>{g.id(v)});
            v=dijkstra_test.predArc(g.source(v));
            //cout << " " << g.id(v);
        }
        while(s!=g.source(v));

        //cout << endl;

        //Insert the found shortest path into the shortest path list.
        shortestPaths.insert(make_pair(dijkstra_test.dist(t),shortestPath));

        int j = nOSlaves;

        //Send the found tasks to all the slaves.
        for(int i=1;i < nOSlaves && !path.empty();i++)
        {
            //Take an element of the task list and then remove it from the list.
            vector<int> tempPath(*path.begin());
            //Send first it's size. It's a simple task and therefore it's size is 1
            int size = 1;
            MPI_Send(&size, 1, MPI_INT, i, 0,	MPI_COMM_WORLD);
            //Now send the list.
            MPI_Send(&*tempPath.begin(), size, MPI_INT, i, 1,	MPI_COMM_WORLD);

            //Fill removedPaths (already sent to slaves) and the runningTasks[i] for each slave.
            removedPaths.push_back(tempPath);
            runningTasks[i] = vector<int>{tempPath};
            path.erase(path.begin());
            --j;
        }

        /**
         * J will increase and decrease depending on how much pending tasks are with the slaves.
         * If the master receives a message he will increase j, if he sends one he will decrease it.
         * If j equals the number of slaves there are no more shortest paths to find.
         */
        while(j<nOSlaves || !path.empty())
        {
            //Create a taskMessage which contains length and distance.
            taskMessage buffer;
            MPI_Status st;

            //First receive the size of the path message
            MPI_Recv(&buffer, sizeof(struct taskMessage), MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &st);
            int i = st.MPI_SOURCE;

            //If the asynchronous message has been received advance, else try again later.
            if (buffer.length > 0)
            {
                //Increase j since the slave terminated a job.
                ++j;
                bool addNew = true;
                //Initialize the taskSize.
                //Initialize the vector to be able to receive the results.
                vector<int> newPath;
                newPath.resize(buffer.length);

                //Receive the result from the slave.
                MPI_Status st1;
                MPI_Recv(&newPath[0], buffer.length, MPI_INT, i, 1, MPI_COMM_WORLD, &st1);
                vector<int> nodePath;

                nodePath.push_back(g.id(g.target(g.arcFromId(*newPath.begin()))));

                for(int arc : newPath)
                {
                    nodePath.push_back(g.id(g.source(g.arcFromId(arc))));
                }

                //If there are more than k paths check if our path is bigger or smaller than one of the given paths.
                if (shortestPaths.size() >= k)
                {
                    //Check if the path already is in the map.
                    typedef multimap<int, vector<int>>::iterator iterator;
                    std::pair<iterator, iterator> iterPair = shortestPaths.equal_range(buffer.distance);
                    bool storedAlready = false;
                    iterator it = iterPair.first;

                    for (; it != iterPair.second; ++it)
                    {
                        if (it->second == nodePath)
                        {
                            storedAlready = true;
                        }
                    }

                    //Get the last object in the map.
                    multimap<int, vector<int>>::const_iterator last = prev(shortestPaths.end());

                    //If it is smaller then add the path to the shortestPaths

                    if (!storedAlready && last->first > buffer.distance)
                    {
                        shortestPaths.erase(last);
                        insertInMap(&shortestPaths, buffer, nodePath);
                    }
                        //Else discard the path and do not add new jobs depending on it.
                    else
                    {
                        addNew = false;
                    }
                }
                else
                {
                    //If less than k paths have been found add the result.
                    insertInMap(&shortestPaths, buffer, nodePath);
                    for(set<vector<int>>::iterator iter = path.begin(); iter != path.end();)
                    {
                        if((*iter).size()>nodePath.size())
                        {
                            path.erase(iter++);
                        }
                        else
                        {
                            ++iter;
                        }
                    }
                }

                //Use the result to add new tasks.

                if (addNew && !runningTasks[i].empty())
                {
                    //The new task will be a cross-product of the result and the runningTask of the slave.
                    //cout << "received message from: " << i << " : ";
                    for (int arc : newPath)
                    {
                        vector<int> nextTask(runningTasks[i]);
                        nextTask.insert(nextTask.begin(), arc);
                        //cout << " " << arc;
                        sort(nextTask.begin(), nextTask.end());
                        if(find(removedPaths.begin(), removedPaths.end(), nextTask) == removedPaths.end())
                            path.insert(nextTask);
                    }
                    //cout << endl;
                }
                //Now remove the task from the slave.
                runningTasks[i].clear();
            }
            else if (buffer.length == -1)
            {
                //If buffer.length == -1 (No path has been found) empty the running task.
                runningTasks[i].clear();
                ++j;
            }

            //Only send messages here - here we know the slave is ready to receive.
            if (!path.empty() && runningTasks[i].empty())
            {
                vector<int> tempPath(*path.begin());
                runningTasks[i] = tempPath;
                int size = tempPath.size();
                MPI_Send(&size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&tempPath[0], size, MPI_INT, i, 1, MPI_COMM_WORLD);
                removedPaths.push_back(tempPath);
                path.erase(path.begin());
                --j;
            }
        }

        cout << "terminating" << endl;

        for ( std::multimap< int, vector<int>, std::less< int > >::const_iterator iter =shortestPaths.begin(); iter != shortestPaths.end(); ++iter )
        {
            cout << iter->first << " : ";
            for(int i=0;i<iter->second.size();i++)
            {
                cout << iter->second[i] << " ";
            }
            cout << endl;
        }

        cout << "shutting down .... " << nOSlaves << endl;
        //send terminate message to all slaves.
        for(int destination=1;destination < nOSlaves && path.empty();destination++)
        {
            int size = 0;
            MPI_Send(&size, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        struct taskMessage task;
        while(true)
        {
            //receive synchronous message with taskSize.
            vector<edgeStorage> removedEdge;
            vector<int> path;
            int size;
            MPI_Status st;
            MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
            //If the size is 0, then the message is a terminate message.
            if (size == 0)
            {
                break;
            }

            //Receive the actual size depending on the task size.
            vector<int> buffer;
            buffer.resize(size);
            MPI_Recv(&buffer[0], size, MPI_INT, 0, 1, MPI_COMM_WORLD, &st);

            //Remove the vertices in order to calculate a new shortest path.
            for (int arc: buffer)
            {
                struct edgeStorage tempEdge;
                tempEdge.arc = g.arcFromId(arc);
                tempEdge.weight = length[g.arcFromId(arc)];
                tempEdge.target = g.target(g.arcFromId(arc));
                //cout << rank << " Removing edge between: " << g.id(g.source(tempEdge.arc)) << " and " << g.id(g.target(tempEdge.arc)) << endl;
                removedEdge.push_back(tempEdge);

                g.changeTarget(g.arcFromId(arc),g.source(g.arcFromId(arc)));
                //g.erase(g.arcFromId(arc));
            }

            //Create an instance of Dijkstra's algorithm and execute it.
            Dijkstra<Graph, LengthMap> dijkstra_test(g, length);
            dijkstra_test.run(s);

            //If the node can't be reached, send this to the master to remove the task.
            if (!dijkstra_test.reached(t))
            {
                //send asynchronous message to master with the new path.
                MPI_Request rq;
                task.length = -1;
                task.distance = 0;
                MPI_Isend(&task, sizeof(struct taskMessage), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &rq);
            }
            else
            {
                //If there is a valid path, print it.
                /*for (Node v = t; v != s; v = dijkstra_test.predNode(v))
                {
                    //cout << g.id(v) << "<-";
                }
                cout << g.id(s) << ": " << dijkstra_test.dist(t) << endl;
                */

                int weight =0;
                //Fill the path with all edges between source s and sink t.
                ListDigraphBase::Arc v=dijkstra_test.predArc(t);
                path.push_back(g.id(v));
                weight += length[v];
                //cout << " " << g.id(v) << "[" << length[v] << "]";
                do
                {
                    v=dijkstra_test.predArc(g.source(v));
                    weight += length[v];

                    path.push_back(g.id(v));
                    //cout << " " << g.id(v) << "[" << length[v] << "]";
                }
                while(s!=g.source(v));
                //cout << " : " << weight << endl;

                //send asynchronous message to master with the new path.
                MPI_Request rq;
                int taskSize = path.size();
                //Send first the resultSize.
                task.length = taskSize;
                task.distance = dijkstra_test.dist(t);
                MPI_Isend(&task, sizeof(struct taskMessage), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &rq);
                //Then send the actual result.
                MPI_Send(&path[0], taskSize, MPI_INT, 0, 1, MPI_COMM_WORLD);
            }

            //Add back the removed edges to the graph.
            for (edgeStorage storage : removedEdge)
            {
                g.changeTarget(storage.arc,storage.target);
                //length[storage.arc] = storage.weight;
                //g.addArc(g.source(storage.arc), g.target(storage.arc));
            }
        }
    }

    MPI_Finalize();

    timestamp_t t11 = get_timestamp();

    long double secs = (t11 - t00) / 1000000.0L;

    cout << secs << endl;
}

/**
 * Main method, which runs the whole program.
 */
int main(int argc,	char	**argv)
{
	run(&argc,&argv);
}
