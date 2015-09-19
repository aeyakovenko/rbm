{-# LANGUAGE BangPatterns #-}
module Data.NN where
 
import qualified Data.Matrix as M
import Control.DeepSeq(NFData)
import Control.Monad(foldM)
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

feedForward :: Monad m => NN -> Matrix U B I -> m (Matrix U B I)
feedForward nn ins = foldM feedForward1 ins nn

backProp :: Monad m => NN -> Double -> Matrix U B I -> Matrix U B Q -> NN 
backProp nn lc ins ebq = do
   outsbi <- scanForwardR ins nn
   let nnjq = map (M.cast1 . M.cast2) nn 

   --output layer backprop
   let doutbi = M.map disigmoid $ head outsbi
   let pbq = (M.cast2 doutbi) ^* ebq
   pqb <- M.transpose pbq

   --hiddel layer backprop results
   pqbs <- scanMR backPropFold pqb $ (zip outsbi (reverse nnjq))

   --apply the backprops
   let pjbs = map M.cast1 $ reverse $ pqb:pqbs
   let apply (wij,pjb,obi) = do
         lji <- (M.cast2 pjb) `M.mmult` obi
         let uji <- M.map ((*) (negate lc)) lji
         uij <- M.transpose uji
         d2u $ wij ^+ uij
   mapM appy (zip nn pjbs outsbi)

--calculate the  backprop for the hidden layers
backPropFold :: Monad m => Matrix U Q B -> (Matrix U B J, Matrix U J Q) -> Matrix U Q B
backPropFold eqb obj wjq = do
   ejb <- wjq `M.mmult` eqb
   dbj <- M.map dsigmoid obj
   djb <- M.transpose dbj
   M.cast1 <$> (M.d2u $ djb *^ ejb)
{-- 
 - compute the error for weights in I,J using error from layer Q
 --}
backProp1 :: Monad m => Matrix U I J -> Matrix U J B -> Matrix U J Q -> Matrix U Q B -> m (Matrix U J B)
backProp1 wij dojb wjq eqb = do
   ejb <- wjq `M.mmult` eqb
   M.d2u $ dojb *^ ejb

scanForwardR :: Monad m => Matrix U B I -> NN -> m ([Matrix U B I])
scanForwardR ins nn = scanMR [] feedForward' ins nn

feedForward1 :: Monad m => Matrix U B I -> Matrix U I J -> m (Matrix U B I)
feedForward1 !ibi wij = do
   sbj <- ibi `M.mmult` wij
   M.cast2 <$> (M.d2u $ M.map sigmoid sbj)

scanMR :: (Monad m) =>  [a] -> (a -> b -> m a) -> a -> [b] -> m [a]
scanMR acc _ _ _ =  return acc
scanMR aac f a ls = do 
   item <- f a (head ls)
   scanMR (item:aac) f item ls


calcWeightUpdate :: Monad m => Matrix U I B -> Matrix U J B -> m (Matrix U I J)
calcWeightUpdate ins ers = ins `M.mmultT` ers

applyWeightUpdate :: Monad m => Double -> Matrix U I J -> Matrix U I J -> m (Matrix U I J)
applyWeightUpdate lr weights update = M.d2u $ (M.map ((*) lr) update) +^ weights

sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))
{-# INLINE sigmoid #-}

dsigmoid:: Double -> Double
dsigmoid s = s * (1 - s)
{-# INLINE dsigmoid #-}


