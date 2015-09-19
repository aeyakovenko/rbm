{-# LANGUAGE BangPatterns #-}
module Data.NN where
 
import qualified Data.Matrix as M
import Control.DeepSeq(NFData)
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
data B -- input batch size

type NN = [Matrix U I J]

feedForward :: Monad m => Matrix U B I -> NN -> m ([Matrix U B I])
feedForward ins nn = scanM feedForward' ins nn

feedForward' :: Monad m => Matrix U B I -> Matrix U I J -> m (Matrix U B I)
feedForward' ibi wij = do
   sbj <- ibi `M.mmult` wij
   M.cast2 <$> (M.d2u $ M.map sigmoid sbj)

scanM :: (NFData a, Monad m) =>  (a -> b -> m a) -> a -> [b] -> m [a]
scanM _ _ [] =  return []
scanM f !a ls = do 
   item <- f a (head ls)
   rest <- scanM f item ls
   return $ item:rest

{-- 
 - compute the error for weights in I,J using error from layer Q
 --}
backProp' :: Monad m => Matrix U I J -> Matrix U J B -> Matrix U J Q -> Matrix U Q B -> m (Matrix U J B)
backProp' wij dojb wjq eqb = do
   ejb <- wjq `M.mmult` eqb
   M.d2u $ dojb *^ ejb

calcWeightUpdate :: Monad m => Matrix U I B -> Matrix U J B -> m (Matrix U I J)
calcWeightUpdate ins ers = ins `M.mmultT` ers

applyWeightUpdate :: Monad m => Double -> Matrix U I J -> Matrix U I J -> m (Matrix U I J)
applyWeightUpdate lr weights update = M.d2u $ (M.map ((*) lr) update) +^ weights

sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))
{-# INLINE sigmoid #-}

dsigmoid:: Double -> Double
dsigmoid d = s * (1 - s)
   where s = sigmoid d
{-# INLINE dsigmoid #-}


