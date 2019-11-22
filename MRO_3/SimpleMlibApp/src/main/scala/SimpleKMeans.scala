import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

object SimpleKMeans {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("KMeansExample").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val data = sc.textFile("./data/kmeans.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    println("Predict " + clusters.predict(Vectors.dense(1f,1f,1f)))
    println("Predict " + clusters.predict(Vectors.dense(8f,8f,8f)))
    val WSSSE = clusters.computeCost(parsedData)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

//    clusters.save(sc, "target/org/apache/spark/KMeansExample/KMeansModel")
//    val sameModel = KMeansModel.load(sc, "target/org/apache/spark/KMeansExample/KMeansModel")
    sc.stop()
  }
}