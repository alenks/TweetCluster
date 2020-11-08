import org.apache.spark.SparkContext
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.feature.{StopWordsRemover, Word2Vec}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.functions.{collect_list, explode, flatten, monotonically_increasing_id}
import org.apache.spark.sql.functions._
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
      .setVectorSize(5)
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
//    predictDf.groupBy("cluster").count().show(100)
    val agg_cluster = predictDf.groupBy("cluster")
      .agg(collect_list("terms").name("terms"))

    val agg_flat_cluster = agg_cluster.select($"cluster", flatten($"terms") alias("terms"))
//    agg_flat_cluster.collect().foreach(println)
    val agg_explode_cluster = agg_flat_cluster.select($"cluster", explode(agg_flat_cluster("terms")) alias("terms"))
    val cluster_wc = agg_explode_cluster.groupBy("cluster", "terms")
      .count()
      .sort($"cluster", $"count".desc)
      .filter("terms != ''")

    println("Most frequent words of each cluster:")
//    cluster_wc.collect().foreach(println)
    val dataWithIndex = cluster_wc.withColumn("idx", monotonically_increasing_id())
//    dataWithIndex.collect().foreach(println)
    val minIdx = dataWithIndex
      .groupBy($"cluster")
      .agg(min($"idx"))
      .sort($"cluster")
      .toDF("r_cluster", "min_idx")
//    minIdx.collect().foreach(println)
//      minIdx.show()

    /*val new_minIdx = minIdx.select($"r_cluster".alias("cluster"), $"min_idx".alias("idx"))

    val cluster_freqw = dataWithIndex.join(
      minIdx,
      ($"cluster" === $"r_cluster") && ($"idx" <= $"min_idx")
    ).select($"cluster", $"terms").sort($"cluster")

    cluster_freqw.collect().foreach(println)*/

    val cluster_freqw = dataWithIndex.join(minIdx)
      .where( $"idx" === $"min_idx" )
      .select($"cluster", $"terms").sort($"cluster")
    //to print
    //cluster_freqw.collect().foreach(println)


//    val cluster_fw = dataWithIndex.join(new_minIdx,Seq("cluster", "idx")).sort("cluster")
//    cluster_fw.collect().foreach(println)

    val evaluator = new ClusteringEvaluator()
      .setFeaturesCol("result")
      .setPredictionCol("cluster")
    val silhouette = evaluator.evaluate(predictDf)
    println(s"Silhouette with squared euclidean distance = $silhouette")
    spark.stop()
  }
}
