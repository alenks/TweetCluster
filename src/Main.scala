import org.apache.spark.SparkContext
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.feature.{StopWordsRemover, Word2Vec}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.sparkproject.dmg.pmml.True

object Main {

  // Using ml.clustering.KMeans (the DataFrame based API)
  def main(args: Array[String]) {
    val spark = SparkSession.builder
      .appName("Simple Application")
      .master("local[*]")
      .getOrCreate()
    val sc = SparkContext.getOrCreate()
    import spark.implicits._

    val inf = sc.textFile(args(0))
    val rawWords = inf.map( x => x.split(" ") ).toDF("text")
    val stopwordList = sc.textFile(args(1)).collect()

    val stopWords = new StopWordsRemover()
      .setInputCol("text")
      .setOutputCol("terms")
      .setCaseSensitive(false)
      .setStopWords(stopwordList)
    val documentDF = stopWords.transform(rawWords)
//    documentDF.show()

    val word2Vec = new Word2Vec()
      .setInputCol("terms")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model = word2Vec.fit(documentDF)

    val result = model.transform(documentDF)
//    result.show()
//    result.collect().foreach(println)
//    result.collect().foreach { case Row(raw: Seq[_], text: Seq[_], features: Vector) =>
//    println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }
    val kmeans = new KMeans()
      .setK(100)
      .setSeed(12L)
      .setFeaturesCol("result")
      .setPredictionCol("cluster")
    val kmmodel = kmeans.fit(result)
    println("Cluster centers:")
    kmmodel.clusterCenters.foreach(println)
    val predictDf = kmmodel.transform(result)
    predictDf.show()
    predictDf.groupBy("cluster").count().show(100)
    val evaluator = new ClusteringEvaluator()
      .setFeaturesCol("result")
      .setPredictionCol("cluster")
    val silhouette = evaluator.evaluate(predictDf)
    println(s"Silhouette with squared euclidean distance = $silhouette")
    spark.stop()
  }
}
