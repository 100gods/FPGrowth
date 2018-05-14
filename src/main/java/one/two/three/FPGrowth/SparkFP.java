package one.two.three.FPGrowth;

import static org.apache.spark.sql.functions.sum;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.split;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.max;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

public class SparkFP {
	private static final Logger LOGGER = Logger.getLogger(SparkFP.class);
	private static final String APP_NAME = "FP Mining";

	public static class AssociationRule implements Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = -3628784869777578718L;
		String rule;
		double Confidence;
		double freqOfX;
		String x;
		double Support;
		int num;

		public String getX() {
			return x;
		}

		public void setX(String x) {
			this.x = x;
		}

		public String getRule() {
			return rule;
		}

		public void setRule(String rule) {
			this.rule = rule;
		}

		public double getConfidence() {
			return Confidence;
		}

		public void setConfidence(double confidence) {
			Confidence = confidence;
		}

		public double getFreqOfX() {
			return freqOfX;
		}

		public void setFreqOfX(double freqOfX) {
			this.freqOfX = freqOfX;
		}

		public double getSupport() {
			return Support;
		}

		public void setSupport(double support) {
			Support = support;
		}

		public int getNum() {
			return num;
		}

		public void setNum(int num) {
			this.num = num;
		}

	}

	public static void main(String[] args) {
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		SparkConf conf = new SparkConf().setMaster("local[4]").setAppName(APP_NAME);
		JavaSparkContext sparkContext = new JavaSparkContext(conf);
		SparkSession session = SparkSession.builder().appName(APP_NAME).config(conf).getOrCreate();
		JavaRDD<String> rdd = sparkContext.textFile("input1.txt");
		JavaRDD<AssociationRule> rulesRdd = rdd.flatMap(new FlatMapFunction<String, AssociationRule>() {
			@Override
			public Iterator<AssociationRule> call(String transaction) throws Exception {
				List<AssociationRule> associationRules = new ArrayList<>();
				String[] items = transaction.split(",");
				List<String> allSets = new ArrayList<>();
				for (int i = 1; i < (1 << items.length) - 1; i++) {
					List<String> temp = new ArrayList<>();
					for (int j = 0; j < items.length; j++) {
						if ((i & (1 << j)) > 0) {
							temp.add(items[j]);
						}
					}
					allSets.add(String.join("|", temp));
				}
				AssociationRule associationRule = null;
				for (String set : allSets) {

					associationRule = new AssociationRule();
					associationRule.setX(set.replaceAll("\\|", ","));
					associationRule.setFreqOfX(1);
					associationRules.add(associationRule);
					for (String item : items) {
						if (set.contains(item)) {
							continue;
						} else {
							associationRule = new AssociationRule();
							associationRule.setRule(set.replaceAll("\\|", ",") + "->" + item);
							associationRule.setConfidence(1);
							// associationRule.setFreqOfX(1);
							associationRule.setSupport(1);
							associationRule.setNum(set.split("\\|").length + 1);
							associationRules.add(associationRule);

						}
					}
				}
				return associationRules.iterator();
			}
		});

		// JavaRDD<String[]> itemsRdd = rdd.map(line -> line.split(","));
		// List<String[]> items = itemsRdd.collect();
		// for (String[] item : items) {
		// for (String strings : item) {
		// System.out.println(strings);
		// }
		// }
		UDF1<String, Integer> splitUdf = new UDF1<String, Integer>() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 7033633417280924362L;

			@Override
			public Integer call(String rule) throws Exception {

				String[] temp = rule.split(",");
				return temp.length + 1;
			}
		};
		session.udf().register("splitUdf", splitUdf, DataTypes.IntegerType);
		Dataset<AssociationRule> ruleDS = session.createDataFrame(rulesRdd, AssociationRule.class)
				.as(Encoders.bean(AssociationRule.class));
		// ruleDS.show(32);
		// ruleDS.printSchema();
		Dataset<Row> freqX = ruleDS.groupBy(ruleDS.col("x")).agg(sum(col("freqOfX")).as("frq(X)"));
		// withColumn("x", callUDF("split", col("rule")))
		Dataset<Row> ruleXY = ruleDS.groupBy(ruleDS.col("rule")).agg(sum(col("confidence")).as("freq(X,Y)"))
				.withColumn("x", split(col("rule"), "->").getItem(0).cast(DataTypes.StringType));
		long N = rdd.count();
		ruleXY = ruleXY.join(freqX, "x").withColumn("N", lit(N).cast(DataTypes.LongType));
		// ruleXY.show();
		ruleXY = ruleXY.withColumnRenamed("rule", "Association Rule")
				.withColumn("Confidence", col("freq(X,Y)").divide(col("frq(X)")))
				.withColumn("Support", col("freq(X,Y)").divide(col("N")));
		ruleXY = ruleXY.withColumn("Number of items in the rule", callUDF("splitUdf", col("Association Rule")));
		ruleXY = ruleXY.select("Association Rule", "Confidence", "Support", "Number of items in the rule");
		Dataset<Row> res = ruleXY.agg(max(col("Confidence")).as("Max Confidence"), max("Support").as("Max Support"),
				max("Number of items in the rule").as("Max Items in any rule"),
				lit(ruleXY.count()).as("Total Num of Rules"));
		res.show();
		ruleXY.show();

	}

}
