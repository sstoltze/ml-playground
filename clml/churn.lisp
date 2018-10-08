(setf *read-default-float-format* 'double-float)

(ql:quickload '(:clml.hjs
                :clml.utility
                :clml.decision-tree))
;(ql:quickload :clml)

(setf lparallel:*kernel* (lparallel:make-kernel 2))

(defun print-dataset (dataset &optional (n 5))
  (let* ((column-names (map 'vector
                            #'clml.hjs.read-data:dimension-name
                            (clml.hjs.read-data:dataset-dimensions dataset)))
         (data-points (coerce (clml.hjs.read-data:head-points dataset
                                                              n)
                                   'list))
         (total-data (cons column-names
                           data-points))
         (width (1+ (reduce #'max
                            (append
                             (map 'list
                                  #'length
                                  column-names)
                             (mapcar #'(lambda (row)
                                         (loop for item across row
                                            maximizing
                                              (length (format nil "~A" item))))
                                     data-points))))))
    (loop for row in total-data
       do
         (loop for item across row
            do (format t "~VA " width item))
         (format t "~%"))))

(defun forest-importance (forest)
  "Stolen from internals in clml.decision-tree.random-forest:importance"
  (labels ((sum-up-decrease-gini (rf-tree column)
             (if (< 2 (length rf-tree))
                 (let ((node-var (caaaar rf-tree))
                       (value (cadaar rf-tree)))
                   (+ (if (string= column node-var)
                          value
                          0.0d0)
                      (sum-up-decrease-gini (second rf-tree) column)
                      (sum-up-decrease-gini (third rf-tree) column)))
                 0.0d0)))
    (loop
       with column-list = (nth 5 (first (aref forest 0)))
       for column in column-list
       as sum-gini = (loop
                        for tree across forest
                        sum (sum-up-decrease-gini tree column))
       collecting (cons column (/ sum-gini (length forest))))))

(defun most-important-features (forest)
    (sort (forest-importance forest)
          #'>=
          :key #'cdr))

(defun forest-accuracy (dataset variable forest)
  (loop for prediction in (clml.decision-tree.random-forest:forest-validation dataset
                                                                              variable
                                                                              forest)
     when (equal (caar prediction) (cdar prediction)) sum (cdr prediction) into correct
     sum (cdr prediction) into total
     finally (return (float (/ correct total)))))


(defun setup-dataset (&optional (path "./Churn-dot-decimal.csv"))
  (clml.hjs.read-data:pick-and-specialize-data
   (clml.hjs.read-data:read-data-from-file
    (clml.utility.data:fetch path)
    :type :csv
    :csv-type-spec '(integer
                     integer
                     double-float
                     double-float
                     double-float
                     double-float
                     integer
                     string
                     string
                     string
                     integer
                     double-float
                     integer
                     double-float
                     integer
                     double-float
                     integer
                     double-float
                     nil
                     nil
                     nil)
    :csv-delimiter #\;)
   :data-types '(:numeric
                 :numeric
                 :numeric
                 :numeric
                 :numeric
                 :numeric
                 :numeric
                 :category
                 :category
                 :category
                 :numeric
                 :numeric
                 :numeric
                 :numeric
                 :numeric
                 :numeric
                 :numeric
                 :numeric)
   :except '(18 19 20)))

(defvar churn-data    (setup-dataset))
(defvar churn-train   nil)
(defvar churn-test    nil)

(multiple-value-setq (churn-train churn-test)
  (clml.hjs.read-data:divide-dataset churn-data
                                     :divide-ratio '(3 1)
                                     :random t))

(defvar random-forest (clml.decision-tree.random-forest:make-random-forest churn-train
                                                                           "Churn"
                                                                           :tree-number 30))

(format t "Features in order of importance:~%")
(loop for feature in (most-important-features random-forest)
   do (format t "~A: ~A~%" (car feature) (cdr feature)))
(format t "~%")
(format t "Prediction on test dataset:~%")
(loop for prediction in (clml.decision-tree.random-forest:forest-validation churn-test
                                                                            "Churn"
                                                                            random-forest)
   as predicted = (format nil "Predicted ~A" (caar prediction))
   as actual = (format nil "actual ~A" (cdar prediction))
   when (equal (caar prediction) (cdar prediction)) sum (cdr prediction) into correct
   sum (cdr prediction) into total
   do (format t "~A, ~A: ~A~%" predicted actual (cdr prediction))
   finally (format t "Correct: ~A, wrong: ~A~%Accuracy: ~A~%" correct (- total correct) (float (/ correct total))))
(format t "~%")
(format t "Accuracy: ~A~%" (forest-accuracy churn-test "Churn" random-forest))
(format t "~%")
