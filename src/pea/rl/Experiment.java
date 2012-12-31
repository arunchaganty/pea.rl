/**
 * pea.rl - Experiment.java.
 *
 * Handles the main event loop between agents and
 * environments.
 */

package pea.rl;

import org.rlcommunity.rlglue.codec.LocalGlue;
import org.rlcommunity.rlglue.codec.RLGlue;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.util.EnvironmentLoader;
import org.rlcommunity.rlglue.codec.util.AgentLoader;
import org.rlcommunity.rlglue.codec.EnvironmentInterface;

import fig.basic.*;
import fig.exec.*;

import java.lang.ClassLoader;

/**
 * Experiment program that handles loading of the agent and environment
 * classes, and running the main RL event loop. 
 *
 * This class also handles some standard logging features.
 */
public class Experiment implements Runnable {

	@Option(gloss = "Path to agent class", required = true)
  public String agentPath;
	@Option(gloss = "Agent arguments" )
  public String agentArgs = "";

	@Option(gloss = "Path to environment class", required = true)
  public String environmentPath;
	@Option(gloss = "Environment arguments")
  public String environmentArgs = "";

	@Option(gloss = "Number of instances the experiment is averaged over")
	public int iters = 1;
	@Option(gloss = "Number of epochs in each instance to run the RL agent")
	public int epochs = 100;
	@Option(gloss = "Maximum number of steps in each epoch")
	public int maxSteps = 5000;
	@Option(gloss = "Whether experiment should be run episodically")
	public boolean isEpisodic = false;

  /**
   * Load from the class path.
   */
  public AgentInterface loadAgent( String agentPath ) 
    throws ClassNotFoundException, InstantiationException, IllegalAccessException {
    ClassLoader classLoader = Experiment.class.getClassLoader();
    Class agentClass = classLoader.loadClass( agentPath );
    return (AgentInterface) agentClass.newInstance();
  }

  /**
   * Load from the class path.
   */
  public EnvironmentInterface loadEnvironment( String envPath ) 
    throws ClassNotFoundException, InstantiationException, IllegalAccessException {
    ClassLoader classLoader = Experiment.class.getClassLoader();
    Class envClass = classLoader.loadClass( envPath );
    return (EnvironmentInterface) envClass.newInstance();
  }

  /**
   * Initialize the agent and environment.
   */
  public void initialize() {
    try {
      // Initialize the agent and environment using class loaders
      AgentInterface agent = loadAgent( agentPath );

      // Initialize the agent and environment using class loaders
      EnvironmentInterface env = loadEnvironment( environmentPath );

      LocalGlue localGlueImplementation = new LocalGlue(env, agent);
      RLGlue.setGlue(localGlueImplementation);

    } catch( ClassNotFoundException | InstantiationException | IllegalAccessException e ) {
      LogInfo.error( e.getMessage() );
      System.exit( 1 );
    }
  }

  /**
   * Print out some average statistics of the run
   */
  public void reportStatistics( double[] avgReturn, double[] avgReturn2 ) {
    assert( avgReturn.length == avgReturn2.length );

    for( int i = 0; i < avgReturn.length; i++ ) {
      System.out.printf( "%d %f %f\n", i, avgReturn[i], Math.sqrt( avgReturn2[i] - avgReturn[i] * avgReturn[i] ) );
    }
  }

  @Override
  public void run() {
    initialize();

    RLGlue.RL_init();

    // Keep track of the average and variance of return 
    double[] avgReturn = new double[ epochs ];
    double[] avgReturn2 = new double[ epochs ];

    RLGlue.RL_agent_message("save_policy policy.dat");
    for( int iter = 0; iter < iters; iter++ ) {
      double ret = 0.0;
      RLGlue.RL_agent_message("load_policy policy.dat");
      for( int epoch = 0; epoch < epochs; epoch++ ) {
        if( isEpisodic ) {
          //RLGlue.RL_episode( maxSteps );
          RLGlue.RL_episode(0);
          ret = RLGlue.RL_return();
        } else {
          double reward = RLGlue.RL_step().r;
          ret += reward;
        }
        avgReturn[epoch] += (ret - avgReturn[epoch])/(iter + 1);
        avgReturn2[epoch] += (ret*ret - avgReturn2[epoch])/(iter + 1);
      }
    }

    reportStatistics(avgReturn, avgReturn2);

    RLGlue.RL_cleanup();
  }

  public static void main( String[] args ) {
		Execution.run( args, new Experiment() );
  }

}


