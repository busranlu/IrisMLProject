using Microsoft.ML;
using System;
using System.IO;

namespace IrisMLProject
{
    class Program
    {
        //data path
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
        //model path, the model will be saved in this path
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "irisClusteringModel.zip");

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed:0);

            //load data
            IDataView dataView = mlContext.Data.LoadFromTextFile<IrisData>(_dataPath, hasHeader: false, separatorChar: ',');

            //learning pipeline
            //KMeans, there are 3 class in dataset 
            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms.Concatenate(featuresColumnName,
                "SepalLength",
                "SepalWidth" ,
                "PetalLength" ,
                "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

            //training
            var model = pipeline.Fit(dataView);
            //saving model to the path that we defined
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }

            //make prediction
            var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);
            var prediction = predictor.Predict(TestIrisData.Setosa);

            
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}" );
            Console.WriteLine($"Distances: {string.Join(" ",prediction.Distances)}" );
            Console.ReadLine();


        }
    }
}
