import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

object Main {
// Using ml.clustering.KMeans (the DataFrame based API)
  def main(args: Array[String]) {
    val spark = SparkSession.builder
      .appName("Simple Application")
      .master("local[*]")
      .getOrCreate()
    val sc = SparkContext.getOrCreate()
    import spark.implicits._

    val input = sc.textFile(args(0)).map(line => line.split(" ").toSeq)
    val word2vec = new Word2Vec()
    val model = word2vec.fit(input)
    val synonyms = model.findSynonyms("1", 5)
    //val stopwords = spark.read.textFile(args(1))


    spark.stop()
  }
}
