module Data.NN where
 
import qualified Data.Matrix as M
import Data.Matrix(Matrix(..)
                  ,(*^)
                  ,(+^)
                  ,U
                  ,D
                  )

-- symbolic data types for matrix dimentions
data I -- number of nodes in row I
data J -- number of nodes in row J
data Q -- number of nodes in row Q
data S -- one dimentional

data NN = NN [Matrix U I J]

sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))
{-# INLINE sigmoid #-}

dsigmoid:: Double -> Double
dsigmoid d = s * (1 - s)
   where s = sigmoid d
{-# INLINE dsigmoid #-}

{-- 
 - compute the error for weights in I,J using error from layer Q
 --}
backProp :: Monad m => Matrix U I J -> Matrix U J S -> Matrix U J Q -> Matrix U Q S -> m (Matrix U J S)
backProp wij dojs wjq eqs = do
   ejs' <- wjq `M.mmult` eqs
   M.d2u $ dojs *^ ejs'

calcWeightUpdate :: Monad m => Matrix U I S -> Matrix U J S -> m (Matrix U I J)
calcWeightUpdate ins ers = ins `M.mmultT` ers

applyWeightUpdate :: Monad m => Double -> Matrix U I J -> Matrix U I J -> m (Matrix U I J)
applyWeightUpdate lr weights update = M.d2u $ (M.map ((*) lr) update) +^ weights

