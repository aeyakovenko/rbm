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
data B -- input batch size

type NN = [Matrix U I J]

feedForward :: Monad m => NN -> Matrix U B I -> m (Matrix U B J)
feedForward nn ins = M.cast2 <$> foldM (M.cast2 . feedForward1) ins nn

backProp :: Monad m => NN -> Double -> Matrix U B I -> Matrix U B J -> m NN 
backProp nn lc ins tbj = do
   outs <- scanForward ins nn
   let routsbj = map M.cast2 $ reverse outs
   let rnn = reverse nn

   --output layer backprop
   ebj <- error (head routsbj) tbj
   pbj <- backPropOutput (head routsbj) ebj

   --hiddel layer backprop results
   pbjs <- scanM (M.cast2 . backPropHidden) pbj (zip (tail routsbj) rnn)

   --apply the backprops
   let fpbjs = reverse pbjs
   mapM (applyBackProp1 lc) (zip nn fpbjs outs)

{--
 - compute error
 --}
error :: Monad m => Matrix U B J -> Matrix U B J -> m Matrix U B J
error obj tbj = d2u $ obj -^ tbj

applyBackProp1 :: Monad m => Double -> (Matrix U I J, Matrix U B J, Matrix U B I) -> m (Matrix U I J)
applyBackPriop1 lc (wij,pbj,obi) = do
   lij <- obi `M.mmultT` (M.cast2 pbj)
   let uij = M.map ((*) (negate lc)) lij
   d2u $ wij ^+ uij

backPropOutput :: Monad m => Matrix U B J -> Matrix U B J -> Matrix U B J
backPropOutput obj ebj = ebj ^* obj

--calculate the  backprop for the hidden layers
backPropHidden :: Monad m => Matrix U B J -> (Matrix U B I, Matrix U I J) -> Matrix U B I
backPropHidden ebq obi wij = do
   ebj <- transpose =<< wij `M.mmultT` ebj
   let dbi = M.map dsigmoid obi
   M.d2u $ dbi *^ (M.cast2 ebj)

scanForward :: Monad m => Matrix U B I -> NN -> m ([Matrix U B J])
scanForward ins nn = M.cast2 <$> scanM (M.cast2 . feedForward1) ins nn

feedForward1 :: Monad m => Matrix U B I -> Matrix U I J -> m (Matrix U B J)
feedForward1 !ibi wij = do
   sbj <- ibi `M.mmult` wij
   M.d2u $ M.map sigmoid sbj

scanM :: (Monad m) =>  (a -> b -> m a) -> a -> [b] -> m [a]
scanM _ a _ =  return [a]
scanM f a ls = do 
   item <- f a (head ls)
   rest <- scanM f item (tail ls)
   return (a:rest)


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


