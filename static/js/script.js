$(document).ready(function () {
    // 
    $("#predictionForm").submit(function (event) {
        event.preventDefault();

        //Jsonify form data
        var dataJson = JSON.stringify( 
            {
                "First_Term_GPA": $('#first_term_gpa').val(),
                "Second_Term_GPA": $('#second_term_gpa').val(),
                "English_Grade": $('#english_grade').val(),
                "First_Language": $('#first_language').val(),
                "Funding": $('#funding').val(),
                "School": $('#school').val(),
                "FastTrack": $('#fast_track').val(),
                "Coop": $('#coop').val(),
                "Residency": $('#residency').val(),
                "Gender": $('#gender').val(),
                "Previous_Education": $('#previous_education').val(),
                "Age_Group": $('#age_group').val()
            }
        );

        // Show the modal and loading spinner
        $("#resultModal").modal("show");
        $("#ModalLongTitle").html("Prediction Result");
        $("#resultLoading").show();
        $("#resultGood").hide().removeClass("animate-scale-up");
        $("#resultSad").hide().removeClass("animate-scale-up");
        $("#resultText").html("Crunching numbers....");

        // Determine the base URL based on the current window's location
        var hostname = window.location.hostname;
        var baseUrl = (hostname === "127.0.0.1" || hostname === "localhost") ? "http://127.0.0.1:5000" : "http://192.168.2.15:5000";

        // Construct the full API URL
        var apiUrl = baseUrl + "/api/predict";

        // Simulate a short delay to wait for API response.
        setTimeout(function () {
            $.ajax({
                url: apiUrl,
                dataType: 'json',
                type: 'POST',
                contentType: 'application/json',
                data: dataJson,
                processData: false,
                success: function (result,status,xhr) {
                    mockPrediction = result.output

                    $("#resultText").html(mockPrediction);
            
                    // Hide the loading spinner and show the appropriate icon and animate it
                    $("#resultLoading").hide();
                    if (mockPrediction === "Student will Persist") {
                        $("#resultGood").show().addClass("animate-scale-up");
                        $("#resultSad").hide();
                    } else {
                        $("#resultSad").show().addClass("animate-scale-up");
                        $("#resultGood").hide();
                    }
                    
                }
            });    
        }, 1000); // 1000 ms (1 second) delay to simulate waiting for the API response

    });

    $("#btn_summary").click(function (event) {
        event.preventDefault();

        // Show the modal and loading spinner
        $("#summaryModal").modal("show");
        $("#summaryModalLongTitle").html("Model Summary");
        $("#summaryresultLoading").show();
        $("#summaryresultText").html("Crunching numbers....");

        // Determine the base URL based on the current window's location
        var hostname = window.location.hostname;
        var baseUrl = (hostname === "127.0.0.1" || hostname === "localhost") ? "http://127.0.0.1:5000" : "http://192.168.2.15:5000";

        // Construct the full API URL
        var apiUrl = baseUrl + "/api/summary";

        // Simulate a short delay (e.g., waiting for API response)
        setTimeout(function () {
            $.ajax({
                url: apiUrl,
                dataType: 'json',
                type: 'GET',
                contentType: 'application/json',
                processData: false,
                success: function (result,status,xhr) {
                    mockPrediction = result.output
                            
                    // Update the result text
                    $("#summaryresultText").html("<b>The summary of the model:</b> <br>" + mockPrediction.replace(/\n/g, "<br>"));

                    $("#summaryresultLoading").hide();

                }
            });        
        }, 1000); // 1000 ms (1 second) delay to simulate waiting for the API response
    });

    $("#btn_scores").click(function (event) {
        event.preventDefault();

        $("#scoreModal").modal("show");
        $("#scoreModalLongTitle").html("Model Score");
        $("#scoreresultLoading").show();
        $("#scoreresultText").html("Crunching numbers....");

        // Determine the base URL based on the current window's location
        var hostname = window.location.hostname;
        var baseUrl = (hostname === "127.0.0.1" || hostname === "localhost") ? "http://127.0.0.1:5000" : "http://192.168.2.15:5000";

        // Construct the full API URL
        var apiUrl = baseUrl + "/api/scores";

        // Simulate a short delay (e.g., waiting for API response)
        setTimeout(function () {
            $.ajax({
                url: apiUrl,
                dataType: 'json',
                type: 'GET',
                contentType: 'application/json',
                processData: false,
                success: function (result,status,xhr) {
                    mockPrediction = result.output

                    // Update the result text
                    $("#scoreresultText").html(
                        "<b>Accuracy Score:</b> " + mockPrediction.accuracy + "% <br>" +
                        "<b>Precision:</b> " + mockPrediction.precision + "%<br>" +
                        "<b>Recall:</b> " + mockPrediction.recall + "%<br>" +
                        "<b>F1:</b> " + mockPrediction.f1 + "%<br>" +
                        "<b>Roc auc:</b> " + mockPrediction.roc_auc + "<br>" +
                        "<b>Confussion Matrix:</b> <br>" + mockPrediction.confussion_matrix.replace(/\n/g, "<br>") + "<br>"
                        //"<b>Test classification_report_var:</b> <br>" + mockPrediction.classification_report_var.replace(/\n/g, "<br>") + "<br>"
                    );
                    $("#scoreresultLoading").hide();

                }
            });        
        }, 1000); // 1000 ms (1 second) delay to simulate waiting for the API response
        
    });

    $('#fillFormButton').click(function () {
        $.ajax({
            url: '/get_random_data',
            type: 'GET',
            success: function(data) {
                // Assuming data contains keys matching the form field IDs.val()

                $('#first_term_gpa').val(data[0])
                $('#second_term_gpa').val(data[1])
                $('#english_grade').val(data[2])
                $('#first_language').val(data[3])
                $('#funding').val(data[4])
                $('#school').val(data[5])
                $('#fast_track').val(data[6])
                $('#coop').val(data[7])
                $('#residency').val(data[8])
                $('#gender').val(data[9])
                $('#previous_education').val(data[10])
                $('#age_group').val(data[11])
            },
            error: function(error) {
                console.log('Error fetching random data:', error);
            }
        });
    });

});